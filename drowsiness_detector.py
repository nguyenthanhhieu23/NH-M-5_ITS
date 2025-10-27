#!/usr/bin/env python3
import cv2
import dlib
import time
import argparse
import numpy as np
import threading
import simpleaudio as sa
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
    args = ap.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = cv2.VideoCapture(args.camera)
    time.sleep(0.5)
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
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 0)
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

            if len(rects) == 0:
                if ALARM_ON:
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
}}}