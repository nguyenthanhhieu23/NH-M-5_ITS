#!/usr/bin/env python3
import cv2
import dlib
import time
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
    ap.add_argument("--debug", action="store_true", help="print diagnostic info when dlib fails")
    ap.add_argument("--enhance", action="store_true", help="apply basic enhancement (CLAHE + unsharp) to improve blurry frames")
    args = ap.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Try to open camera using DirectShow on Windows to avoid MSMF/MediaFoundation errors
    preferred_backends = []
    if platform.system() == 'Windows':
        preferred_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        preferred_backends = [cv2.CAP_ANY]

    vs = None
    opened_backend = None
    for backend in preferred_backends:
        try:
            vs = cv2.VideoCapture(args.camera, backend)
            time.sleep(0.5)
            if vs is not None and vs.isOpened():
                opened_backend = backend
                break
            # release and try next
            if vs is not None:
                vs.release()
        except Exception:
            # try next backend
            pass

    if vs is None or not vs.isOpened():
        print(f"[FATAL] Unable to open camera index {args.camera} with backends {preferred_backends}")
        print("- Close other apps using the camera, try different --camera index, or check Windows camera privacy settings/drivers.")
        sys.exit(1)

    backend_names = {cv2.CAP_DSHOW: 'CAP_DSHOW', cv2.CAP_MSMF: 'CAP_MSMF', cv2.CAP_ANY: 'CAP_ANY'}
    print(f"[INFO] Opened camera index {args.camera} using backend {backend_names.get(opened_backend, str(opened_backend))}")

    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vs.get(cv2.CAP_PROP_FPS) or 20.0

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    COUNTER = 0
    ALARM_ON = False
    alarm = AlarmPlayer(freq=1200, duration_ms=400)
    ear_history = deque(maxlen=10)

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                print("[WARNING] Không đọc được frame từ camera")
                continue

            # Ensure frame is 8-bit and C-contiguous before conversion
            if frame.dtype != np.uint8:
                # convertScaleAbs handles float -> uint8 safely
                frame = cv2.convertScaleAbs(frame)
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.dtype != np.uint8 or not gray.flags['C_CONTIGUOUS']:
                gray = np.ascontiguousarray(gray, dtype=np.uint8)

            # Call detector; if dlib rejects the image type, try RGB fallback and print diagnostics
            try:
                rects = detector(gray, 0)
            except RuntimeError as e:
                print(f"[ERROR] dlib detector error on gray image: {e}")
                print(f"        gray.dtype={gray.dtype}, shape={gray.shape}, contiguous={gray.flags['C_CONTIGUOUS']}")
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if rgb.dtype != np.uint8:
                        rgb = cv2.convertScaleAbs(rgb)
                    if not rgb.flags['C_CONTIGUOUS']:
                        rgb = np.ascontiguousarray(rgb)
                    rects = detector(rgb, 0)
                    print("[INFO] detector succeeded on RGB fallback")
                except Exception as e2:
                    print(f"[ERROR] dlib detector error on RGB fallback: {e2}")
                    rects = []

            for rect in rects:
                shape = predictor(gray, rect)
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

                cv2.putText(frame, f"EAR: {smooth_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Frames: {COUNTER}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if len(rects) == 0 and ALARM_ON:
                ALARM_ON = False
                alarm.stop()

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Drowsiness Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

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