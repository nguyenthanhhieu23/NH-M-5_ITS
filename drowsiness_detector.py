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
    frame_idx = 0
    
    # Thêm biến để theo dõi thời gian dừng alarm
    ALARM_STOP_TIME = 0
    ALARM_COOLDOWN_SECONDS = 5  # Chờ 5 giây trước khi cho phép alarm bắt đầu lại

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
                    # Kiểm tra cooldown trước khi cho phép alarm bắt đầu
                    current_time = time.time()
                    can_start_alarm = (current_time - ALARM_STOP_TIME) > ALARM_COOLDOWN_SECONDS
                    
                    if COUNTER >= args.ear_consec_frames and can_start_alarm:
                        if not ALARM_ON:
                            ALARM_ON = True
                            alarm.start()
                            print("[ALARM] Phát hiện buồn ngủ - Alarm bắt đầu!")
                        # Tiếp tục hiển thị cảnh báo khi alarm đang ON
                        overlay = frame.copy()
                        alpha = 0.4
                        cv2.rectangle(overlay, (0,0), (width, height), (0,0,255), -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        cv2.putText(frame, "WAKE UP! DROWSINESS DETECTED", (10, height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                else:
                    # Khi EAR > ngưỡng, vẫn giữ alarm ON nếu đã được kích hoạt
                    if ALARM_ON:
                        # Tiếp tục hiển thị cảnh báo ngay cả khi EAR > ngưỡng
                        overlay = frame.copy()
                        alpha = 0.3  # Giảm độ mờ một chút
                        cv2.rectangle(overlay, (0,0), (width, height), (0,0,255), -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        cv2.putText(frame, "STILL DROWSY - Press 'S' to stop", (10, height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.putText(frame, f"EAR: {smooth_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Frames: {COUNTER}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                # Hiển thị trạng thái alarm chi tiết
                if ALARM_ON:
                    cv2.putText(frame, "ALARM: ON - Press 'S' to stop", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    cv2.putText(frame, f"EAR Threshold: {args.ear_thresh}", 
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    current_time = time.time()
                    time_since_stop = current_time - ALARM_STOP_TIME
                    if time_since_stop < ALARM_COOLDOWN_SECONDS:
                        remaining_time = ALARM_COOLDOWN_SECONDS - time_since_stop
                        cv2.putText(frame, f"ALARM: COOLDOWN ({remaining_time:.1f}s)", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
                    else:
                        cv2.putText(frame, "ALARM: READY", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, f"Need {args.ear_consec_frames} frames to trigger", 
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Không tự động dừng alarm khi không phát hiện khuôn mặt
            # Alarm sẽ tiếp tục cho đến khi người dùng can thiệp

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
            elif key == ord('a'):
                if ALARM_ON:
                    ALARM_ON = False
                    alarm.stop()
                else:
                    ALARM_ON = True
                    alarm.start()
            # handle 's' key for both stopping alarm and saving snapshot
            elif key == ord("s"):
                if ALARM_ON:
                    # Dừng alarm thủ công
                    ALARM_ON = False
                    alarm.stop()
                    COUNTER = 0  # Reset counter khi dừng alarm thủ công
                    ALARM_STOP_TIME = time.time()  # Ghi nhận thời gian dừng alarm
                    print("[INFO] Alarm đã được dừng thủ công bởi người dùng")
                
                # Save snapshot if save_dir is configured
                if args.save_dir:
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