#!/usr/bin/env python3
import cv2
import dlib
import time
import os
import argparse
import numpy as np
import threading
import simpleaudio as sa
import sys
import platform
from imutils import face_utils
from collections import deque

def euclidean(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

class AlarmPlayer:
    def __init__(self, freq=1000, duration_ms=500):
        self.freq = freq
        self.duration_ms = duration_ms
        self._stop_event = threading.Event()
        self._thread = None

    def _make_tone(self):
        fs = 44100
        t = np.linspace(0, self.duration_ms / 1000.0, int(fs * (self.duration_ms / 1000.0)), False)
        tone = 0.5 * np.sin(2 * np.pi * self.freq * t)
        audio = (tone * (2**15 - 1)).astype(np.int16)
        return audio

    def _play_loop(self):
        wave = self._make_tone()
        while not self._stop_event.is_set():
            try:
                sa.play_buffer(wave, 1, 2, 44100).wait_done()
            except Exception:
                time.sleep(self.duration_ms / 1000.0)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._play_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape-predictor", required=True, help="path to dlib's 68-point shape predictor")
    ap.add_argument("--camera", type=int, default=0, help="camera device index")
    ap.add_argument("--ear-thresh", type=float, default=0.25, help="EAR threshold to consider eye closed")
    ap.add_argument("--ear-consec-frames", type=int, default=20, help="consecutive frames threshold for alarm")
    ap.add_argument("--output", default=None, help="optional: output video file (ex: out.avi)")
    ap.add_argument("--save-dir", default=None, help="optional: directory to save captured images")
    ap.add_argument("--save-all", action="store_true", help="save every captured frame to --save-dir")
    ap.add_argument("--save-on-alarm", action="store_true", help="save frames only when alarm is ON")
    ap.add_argument("--save-interval", type=int, default=1, help="save every N frames (default 1)")
    ap.add_argument("--debug", action="store_true", help="print diagnostic info when dlib fails")
    ap.add_argument("--enhance", action="store_true", help="apply basic enhancement (CLAHE + unsharp) to improve blurry frames")
    ap.add_argument("--force-alarm", action="store_true", help="start alarm immediately and keep it on until toggled")
    ap.add_argument("--save-once-per-event", action="store_true", help="save a single snapshot when alarm first turns ON")
    args = ap.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

   #  nếu bộ phát hiện dlib liên tục thất bại trên hệ thống này, hãy sử dụng OpenCV Haar cascade
    cascade = None
    cascade_enabled = False
    dlib_failures = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Try multiple camera indices and backends to find a working capture device
    preferred_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if platform.system() == 'Windows' else [cv2.CAP_ANY]
    vs = None
    opened_backend = None
    opened_index = None

    # Try requested index first, then a few nearby indices
    candidate_indices = [args.camera] + [i for i in range(0, 4) if i != args.camera]

    tried = []
    for idx in candidate_indices:
        for backend in preferred_backends:
            try:
                if backend == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend)
                time.sleep(0.2)
                ok = cap is not None and cap.isOpened()
                tried.append((idx, backend, ok))
                if ok:
                    vs = cap
                    opened_backend = backend
                    opened_index = idx
                    break
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
            except Exception:
                pass
        if vs is not None:
            break

    if vs is None or not vs.isOpened():
        print(f"[FATAL] Unable to open any camera. Tried: {[(i,b) for (i,b,ok) in tried if not ok]}")
        print("- Close other apps using the camera, try a different --camera index, check Windows camera privacy settings/drivers, or try a different USB port/webcam.")
        sys.exit(1)

    backend_names = {cv2.CAP_DSHOW: 'CAP_DSHOW', cv2.CAP_MSMF: 'CAP_MSMF', cv2.CAP_ANY: 'CAP_ANY'}
    print(f"[INFO] Opened camera index {opened_index} using backend {backend_names.get(opened_backend, str(opened_backend))}")

    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vs.get(cv2.CAP_PROP_FPS) or 20.0

    # Failure/reopen controls: try to recover automatically when frame grabs fail repeatedly
    consecutive_failed_reads = 0
    max_failed_reads_reopen = 15
    reopen_attempts = 0
    max_reopen_attempts = 4
    reopen_delay_sec = 1.0
    # Try reducing resolution if reopening helps stability
    lower_resolutions = [(640, 480), (320, 240)]

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    COUNTER = 0
    ALARM_ON = False
    alarm = AlarmPlayer(freq=1200, duration_ms=400)
    ear_history = deque(maxlen=10)
    frame_idx = 0
    # track whether we've saved a snapshot for the current alarm event
    saved_on_current_alarm = False

    # Prepare save directory if requested
    if args.save_dir:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
        except Exception as e:
            print(f"[FATAL] Cannot create save directory {args.save_dir}: {e}")
            sys.exit(1)

    # If user wants the alarm forced on, start it now
    if args.force_alarm:
        ALARM_ON = True
        alarm.start()

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                consecutive_failed_reads += 1
                print(f"[WARNING] Không đọc được frame từ camera (consecutive failures: {consecutive_failed_reads})")
                # small backoff to avoid busy loop
                time.sleep(0.1)

                # If repeated failures, try to recover by releasing & reopening the capture
                if consecutive_failed_reads >= max_failed_reads_reopen:
                    reopen_attempts += 1
                    if reopen_attempts > max_reopen_attempts:
                        print(f"[FATAL] Camera failed after {reopen_attempts} reopen attempts. Exiting loop.")
                        break

                    print("[INFO] Attempting to reopen camera to recover from repeated frame grab failures...")
                    try:
                        vs.release()
                    except Exception:
                        pass

                    reopened = False
                    tried2 = []
                    # try the candidate indices and backends again
                    for idx in candidate_indices:
                        for backend in preferred_backends:
                            try:
                                if backend == cv2.CAP_ANY:
                                    cap = cv2.VideoCapture(idx)
                                else:
                                    cap = cv2.VideoCapture(idx, backend)
                                time.sleep(0.3)
                                ok = cap is not None and cap.isOpened()
                                tried2.append((idx, backend, ok))
                                if ok:
                                    vs = cap
                                    opened_backend = backend
                                    opened_index = idx
                                    reopened = True
                                    break
                                else:
                                    try:
                                        cap.release()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        if reopened:
                            break

                    if not reopened:
                        print(f"[WARN] Reopen attempt failed (tried: {[(i,b) for (i,b,ok) in tried2 if not ok]}); will retry after delay.")
                        time.sleep(reopen_delay_sec)
                        continue
                    else:
                        print(f"[INFO] Reopened camera index {opened_index} using backend {backend_names.get(opened_backend, str(opened_backend))}")
                        # try lowering resolution to improve stability
                        for (w, h) in lower_resolutions:
                            try:
                                vs.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                                vs.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                                time.sleep(0.2)
                                nw = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
                                nh = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                if nw == w and nh == h:
                                    width, height = nw, nh
                                    break
                            except Exception:
                                pass
                        # reset counters after successful reopen
                        consecutive_failed_reads = 0
                        continue

            # on successful read reset consecutive failure counters
            consecutive_failed_reads = 0
            reopen_attempts = 0

            # Ensure frame is 8-bit and C-contiguous before conversion
            if frame.dtype != np.uint8:
                # convertScaleAbs handles float -> uint8 safely
                frame = cv2.convertScaleAbs(frame)
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.dtype != np.uint8 or not gray.flags['C_CONTIGUOUS']:
                gray = np.ascontiguousarray(gray, dtype=np.uint8)

            # Prepare an RGB copy for dlib predictor (some dlib builds prefer RGB numpy arrays)
            try:
                rgb_for_predictor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rgb_for_predictor.dtype != np.uint8:
                    rgb_for_predictor = cv2.convertScaleAbs(rgb_for_predictor)
                if not rgb_for_predictor.flags['C_CONTIGUOUS']:
                    rgb_for_predictor = np.ascontiguousarray(rgb_for_predictor)
            except Exception:
                rgb_for_predictor = None

            # Try dlib detector first (preferred). If it fails repeatedly, switch to OpenCV cascade.
            rects = []
            if not cascade_enabled:
                try:
                    rects = detector(gray, 0)
                except RuntimeError as e:
                    dlib_failures += 1
                    if args.debug:
                        print(f"[ERROR] dlib detector error on gray image: {e}")
                        print(f"        gray.dtype={gray.dtype}, shape={gray.shape}, contiguous={gray.flags['C_CONTIGUOUS']}")
                    # try RGB once
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if rgb.dtype != np.uint8:
                            rgb = cv2.convertScaleAbs(rgb)
                        if not rgb.flags['C_CONTIGUOUS']:
                            rgb = np.ascontiguousarray(rgb)
                        rects = detector(rgb, 0)
                        if args.debug:
                            print("[INFO] detector succeeded on RGB fallback")
                    except Exception as e2:
                        if args.debug:
                            print(f"[ERROR] dlib detector error on RGB fallback: {e2}")
                        rects = []

                # If dlib has failed multiple times, enable OpenCV Haar cascade fallback
                if dlib_failures >= 3 and not cascade_enabled:
                    try:
                        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        cascade = cv2.CascadeClassifier(cascade_path)
                        if cascade.empty():
                            if args.debug:
                                print(f"[WARN] Could not load cascade at {cascade_path}")
                        else:
                            cascade_enabled = True
                            if args.debug:
                                print("[INFO] Enabled OpenCV Haar cascade fallback for face detection")
                    except Exception as e:
                        if args.debug:
                            print(f"[WARN] Exception while loading cascade: {e}")
            # If cascade fallback is enabled, use it to detect faces
            if cascade_enabled:
                try:
                    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    rects = [dlib.rectangle(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]
                except Exception as e:
                    if args.debug:
                        print(f"[WARN] Cascade detection failed: {e}")
                    rects = []

            for rect in rects:
                # prefer RGB image for the predictor; fall back to gray if it errors
                shape = None
                if rgb_for_predictor is not None:
                    try:
                        shape = predictor(rgb_for_predictor, rect)
                    except RuntimeError as e:
                        if args.debug:
                            print(f"[WARN] predictor failed on RGB image: {e}; will try gray image")
                        shape = None
                if shape is None:
                    try:
                        shape = predictor(gray, rect)
                    except RuntimeError as e:
                        # give a clear debug message and skip this rect
                        if args.debug:
                            print(f"[ERROR] predictor failed on gray image: {e}")
                        continue
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                ear_history.append(ear)
                smooth_ear = np.mean(ear_history)

                leftHull = cv2.convexHull(leftEye)
                rightHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

                if smooth_ear < args.ear_thresh:
                    COUNTER += 1
                    if COUNTER >= args.ear_consec_frames:
                        if not ALARM_ON:
                            ALARM_ON = True
                            alarm.start()
                            # Save once when alarm first turns on, if requested
                            if args.save_once_per_event and args.save_dir and not saved_on_current_alarm:
                                try:
                                    from datetime import datetime
                                    fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_alarm.jpg"
                                    path = os.path.join(args.save_dir, fn)
                                    cv2.imwrite(path, frame)
                                    saved_on_current_alarm = True
                                    if args.debug:
                                        print(f"[INFO] Saved alarm snapshot to {path}")
                                except Exception as e:
                                    print(f"[WARN] Failed to save alarm snapshot: {e}")
                        overlay = frame.copy()
                        alpha = 0.4
                        cv2.rectangle(overlay, (0,0), (width, height), (0,0,255), -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        cv2.putText(frame, "WAKE UP! DROWSINESS DETECTED", (10, height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                else:
                    COUNTER = 0
                    if ALARM_ON:
                        ALARM_ON = False
                        alarm.stop()
                        # reset the per-event saved flag when alarm stops
                        saved_on_current_alarm = False

                cv2.putText(frame, f"EAR: {smooth_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Frames: {COUNTER}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if len(rects) == 0 and ALARM_ON:
                ALARM_ON = False
                alarm.stop()
                # reset the per-event saved flag when alarm stops (no faces)
                saved_on_current_alarm = False

            if writer is not None:
                writer.write(frame)

            # Save frames according to options
            frame_idx += 1
            save_this = False
            if args.save_dir:
                if args.save_all:
                    if frame_idx % max(1, args.save_interval) == 0:
                        save_this = True
                elif args.save_on_alarm and ALARM_ON:
                    if frame_idx % max(1, args.save_interval) == 0:
                        save_this = True

            if save_this:
                from datetime import datetime
                fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
                path = os.path.join(args.save_dir, fn)
                try:
                    cv2.imwrite(path, frame)
                    if args.debug:
                        print(f"[INFO] Saved frame to {path}")
                except Exception as e:
                    print(f"[WARN] Failed to save frame to {path}: {e}")

            cv2.imshow("Drowsiness Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # toggle alarm manually with 'a'
            if key == ord('a'):
                if ALARM_ON:
                    ALARM_ON = False
                    alarm.stop()
                else:
                    ALARM_ON = True
                    alarm.start()
            # manual save snapshot with 's'
            if key == ord('s') and args.save_dir:
                from datetime import datetime
                fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_manual.jpg"
                path = os.path.join(args.save_dir, fn)
                try:
                    cv2.imwrite(path, frame)
                    if args.debug:
                        print(f"[INFO] Saved manual snapshot to {path}")
                except Exception as e:
                    print(f"[WARN] Failed to save manual snapshot to {path}: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        alarm.stop()
        if writer is not None:
            writer.release()
        vs.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()