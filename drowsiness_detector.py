#!/usr/bin/env python3
import cv2
import dlib
import time
import os
import argparse
import numpy as np
import threading
import wave
import tempfile
import subprocess
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

    def _write_wave_file(self, audio, fs):
        # Ghi mảng âm thanh numpy int16 ra tệp WAV tạm thời và trả về đường dẫn
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp_name = tmp.name
        tmp.close()
        with wave.open(tmp_name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())
        return tmp_name

    def _play_with_subprocess(self, wav_path):
        # Phát âm thanh bằng trình phát mặc định của hệ điều hành để tránh lỗi C-extension
        try:
            if platform.system() == 'Darwin':
                # macOS
                subprocess.run(['afplay', wav_path], check=False)
            elif platform.system() == 'Linux':
                subprocess.run(['aplay', wav_path], check=False)
            elif platform.system() == 'Windows':
                # Sử dụng PowerShell SoundPlayer
                cmd = ['powershell', '-c', "(New-Object Media.SoundPlayer '{0}').PlaySync()".format(wav_path)]
                subprocess.run(cmd, check=False)
            else:
                # Dự phòng: thử afplay/aplay
                subprocess.run(['afplay', wav_path], check=False)
        except Exception:
            # Bỏ qua lỗi phát âm thanh (nếu có)
            pass

    def _play_loop(self):
        fs = 44100
        audio = self._make_tone()
        wav_path = None
        try:
            wav_path = self._write_wave_file(audio, fs)
            while not self._stop_event.is_set():
                self._play_with_subprocess(wav_path)
                # Nghỉ ngắn để tránh vòng lặp chạy quá nhanh nếu trình phát kết thúc sớm
                time.sleep(max(0.01, self.duration_ms / 1000.0))
        finally:
            try:
                if wav_path is not None and os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass

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
    ap.add_argument("--shape-predictor", required=True, help="đường dẫn đến file shape predictor 68 điểm của dlib")
    ap.add_argument("--camera", type=int, default=0, help="chỉ số thiết bị camera")
    ap.add_argument("--ear-thresh", type=float, default=0.25, help="ngưỡng EAR để xác định mắt nhắm")
    ap.add_argument("--ear-consec-frames", type=int, default=20, help="số khung hình liên tiếp để kích hoạt cảnh báo")
    ap.add_argument("--open-consec-frames", type=int, default=3, help="số khung mở mắt liên tiếp để tự tắt cảnh báo")
    ap.add_argument("--output", default=None, help="tùy chọn: tệp video đầu ra (ví dụ: out.avi)")
    ap.add_argument("--save-dir", default=None, help="tùy chọn: thư mục lưu ảnh chụp")
    ap.add_argument("--save-all", action="store_true", help="lưu mọi khung hình vào thư mục --save-dir")
    ap.add_argument("--save-on-alarm", action="store_true", help="chỉ lưu khung hình khi cảnh báo đang bật")
    ap.add_argument("--save-interval", type=int, default=1, help="lưu mỗi N khung hình (mặc định 1)")
    ap.add_argument("--debug", action="store_true", help="in thông tin chẩn đoán khi dlib lỗi")
    ap.add_argument("--enhance", action="store_true", help="tăng cường hình ảnh (CLAHE + unsharp) để cải thiện khung mờ")
    ap.add_argument("--force-alarm", action="store_true", help="bật cảnh báo ngay lập tức cho đến khi tắt")
    args = ap.parse_args()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Thử mở camera bằng DirectShow trên Windows để tránh lỗi MSMF/MediaFoundation
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
            # Giải phóng và thử backend khác
            if vs is not None:
                vs.release()
        except Exception:
            # Thử backend kế tiếp
            pass

    if vs is None or not vs.isOpened():
        print(f"[FATAL] Không thể mở camera index {args.camera} với backend {preferred_backends}")
        print("- Đóng các ứng dụng đang dùng camera, thử chỉ số khác, hoặc kiểm tra quyền truy cập camera.")
        sys.exit(1)

    backend_names = {cv2.CAP_DSHOW: 'CAP_DSHOW', cv2.CAP_MSMF: 'CAP_MSMF', cv2.CAP_ANY: 'CAP_ANY'}
    print(f"[INFO] Đã mở camera index {args.camera} với backend {backend_names.get(opened_backend, str(opened_backend))}")

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
    OPEN_COUNTER = 0
    
    # Thêm biến theo dõi thời gian dừng cảnh báo
    ALARM_STOP_TIME = 0
    ALARM_COOLDOWN_SECONDS = 0.1  # Chờ 0,1 giây trước khi cho phép bật lại cảnh báo

    # Chuẩn bị thư mục lưu ảnh (nếu có yêu cầu)
    if args.save_dir:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
        except Exception as e:
            print(f"[FATAL] Không thể tạo thư mục lưu ảnh {args.save_dir}: {e}")
            sys.exit(1)

    # Nếu người dùng yêu cầu bật cảnh báo cưỡng bức
    if args.force_alarm:
        ALARM_ON = True
        alarm.start()

    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                print("[WARNING] Không đọc được khung hình từ camera")
                continue

            # Đảm bảo frame ở định dạng 8-bit và liên tục trong bộ nhớ
            if frame.dtype != np.uint8:
                frame = cv2.convertScaleAbs(frame)
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.dtype != np.uint8 or not gray.flags['C_CONTIGUOUS']:
                gray = np.ascontiguousarray(gray, dtype=np.uint8)

            # Gọi detector; nếu dlib từ chối loại ảnh, thử lại với RGB và in thông tin chẩn đoán
            try:
                rects = detector(gray, 0)
            except RuntimeError as e:
                print(f"[ERROR] Lỗi dlib detector trên ảnh xám: {e}")
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if rgb.dtype != np.uint8:
                        rgb = cv2.convertScaleAbs(rgb)
                    if not rgb.flags['C_CONTIGUOUS']:
                        rgb = np.ascontiguousarray(rgb)
                    rects = detector(rgb, 0)
                    print("[INFO] Detector thành công khi chuyển sang RGB")
                except Exception as e2:
                    print(f"[ERROR] Lỗi detector trên RGB fallback: {e2}")
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
                    OPEN_COUNTER = 0
                    # Kiểm tra thời gian cooldown trước khi bật cảnh báo
                    current_time = time.time()
                    can_start_alarm = (current_time - ALARM_STOP_TIME) > ALARM_COOLDOWN_SECONDS
                    
                    if COUNTER >= args.ear_consec_frames and can_start_alarm:
                        if not ALARM_ON:
                            ALARM_ON = True
                            alarm.start()
                            print("[ALARM] Phát hiện buồn ngủ - Bắt đầu cảnh báo!")
                        # Hiển thị cảnh báo khi cảnh báo đang bật
                        overlay = frame.copy()
                        alpha = 0.4
                        cv2.rectangle(overlay, (0,0), (width, height), (0,0,255), -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        cv2.putText(frame, "THUC DAY! PHAT HIEN BUON NGU", (10, height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                else:
                    # Khi EAR >= ngưỡng: tăng bộ đếm mở mắt liên tiếp
                    OPEN_COUNTER += 1
                    # Nếu mắt mở đủ số khung và cảnh báo đang bật → tự tắt cảnh báo
                    if ALARM_ON and OPEN_COUNTER >= args.open_consec_frames:
                        ALARM_ON = False
                        alarm.stop()
                        COUNTER = 0
                        ALARM_STOP_TIME = time.time()
                        OPEN_COUNTER = 0
                        if args.debug:
                            print("[INFO] Cảnh báo tự dừng do mắt mở liên tục")
                    # Nếu cảnh báo vẫn đang bật, tiếp tục hiển thị overlay
                    if ALARM_ON:
                        overlay = frame.copy()
                        alpha = 0.3
                        cv2.rectangle(overlay, (0,0), (width, height), (0,0,255), -1)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        cv2.putText(frame, "VAN BUON NGU - Nhan 'S' de dung", (10, height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.putText(frame, f"EAR: {smooth_ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Khung: {COUNTER}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                # Hiển thị trạng thái cảnh báo chi tiết
                if ALARM_ON:
                    cv2.putText(frame, "CANH BAO: BAT - Nhan 'S' de dung", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    cv2.putText(frame, f"Ngưỡng EAR: {args.ear_thresh}", 
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    current_time = time.time()
                    time_since_stop = current_time - ALARM_STOP_TIME
                    if time_since_stop < ALARM_COOLDOWN_SECONDS:
                        remaining_time = ALARM_COOLDOWN_SECONDS - time_since_stop
                        cv2.putText(frame, f"CANH BAO: DANG CHO ({remaining_time:.1f}s)", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,165,0), 2)
                    else:
                        cv2.putText(frame, "CANH BAO: SAN SANG", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.putText(frame, f"Can {args.ear_consec_frames} khung de kich hoat", 
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Không tự động dừng cảnh báo khi không phát hiện khuôn mặt
            # Cảnh báo sẽ tiếp tục cho đến khi người dùng tắt

            if writer is not None:
                writer.write(frame)

            # Lưu khung hình theo tùy chọn người dùng
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
                        print(f"[INFO] Đã lưu khung hình tại {path}")
                except Exception as e:
                    print(f"[WARN] Lỗi khi lưu khung hình {path}: {e}")

            cv2.imshow("Drowsiness Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            # Nhấn 'a' để bật/tắt cảnh báo thủ công
            elif key == ord('a'):
                if ALARM_ON:
                    ALARM_ON = False
                    alarm.stop()
                else:
                    ALARM_ON = True
                    alarm.start()
            # Nhấn 's' để dừng cảnh báo hoặc chụp ảnh
            elif key == ord("s"):
                if ALARM_ON:
                    # Dừng cảnh báo thủ công
                    ALARM_ON = False
                    alarm.stop()
                    COUNTER = 0
                    ALARM_STOP_TIME = time.time()
                    print("[INFO] Cảnh báo đã được dừng thủ công")
                
                # Lưu ảnh chụp nếu có thư mục lưu
                if args.save_dir:
                    from datetime import datetime
                    fn = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_manual.jpg"
                    path = os.path.join(args.save_dir, fn)
                    try:
                        cv2.imwrite(path, frame)
                        if args.debug:
                            print(f"[INFO] Đã lưu ảnh thủ công tại {path}")
                    except Exception as e:
                        print(f"[WARN] Lỗi khi lưu ảnh thủ công {path}: {e}")

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
