# Hệ thống phát hiện buồn ngủ cho tài xế (Dự án ITS)

## 🧠 Tổng quan
Hệ thống phát hiện buồn ngủ của tài xế theo thời gian thực bằng webcam.  
Sử dụng **dlib (68 điểm landmark trên khuôn mặt)** để tính Eye Aspect Ratio (EAR) và phát cảnh báo khi mắt nhắm trong một số khung hình liên tiếp.

---

## ⚙️ Yêu cầu
- Python 3.8+
- Cài đặt các phụ thuộc:
  ```bash
  pip install -r requirements.txt
  ```

Nếu `dlib` cài đặt không thành công, thử:
```bash
pip install cmake
pip install dlib
```
hoặc sử dụng Anaconda:
```bash
conda install -c conda-forge dlib
```

---

## 📦 Nội dung thư mục
```
drowsiness_detector.py
requirements.txt
README.md
```

Bạn cần thêm file:
**shape_predictor_68_face_landmarks.dat**
Tải từ:
https://github.com/davisking/dlib-models
(đặt cùng thư mục với script)

---

## ▶️ Chạy chương trình

### Cách 1: Sử dụng virtual environment (khuyến nghị)
```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Chạy chương trình
python drowsiness_detector.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

### Cách 2: Sử dụng python3 trực tiếp
```bash
python3 drowsiness_detector.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

### Các tùy chọn thông thường
| Tham số | Mô tả | Mặc định |
|--------:|:------|:--------|
| `--camera` | Chỉ số camera | 0 |
| `--ear-thresh` | Ngưỡng EAR để xác định mắt nhắm | 0.25 |
| `--ear-consec-frames` | Số khung hình liên tiếp để kích hoạt cảnh báo | 20 |
| `--open-consec-frames` | Số khung hình mở mắt liên tiếp để tự dừng cảnh báo | 3 |
| `--output` | Ghi video đầu ra (ví dụ out.avi) | None |
| `--save-dir` | Thư mục để lưu ảnh/chụp khung hình | None |
| `--save-all` | Lưu mọi khung hình vào `--save-dir` | False |
| `--save-on-alarm` | Chỉ lưu khi cảnh báo bật | False |
| `--save-interval` | Lưu mỗi N khung hình | 1 |
| `--debug` | In thông tin chẩn đoán khi dlib lỗi | False |
| `--enhance` | Tăng cường ảnh (CLAHE + unsharp) | False |
| `--force-alarm` | Bật cảnh báo ngay khi khởi động | False |
| `--tts` | Sử dụng TTS hệ thống để nói cảnh báo | False |
| `--alarm-tts-text` | Nội dung TTS khi `--tts` bật | "Thức dậy, phát hiện buồn ngủ" |

### Phím điều khiển
- Nhấn `q` để thoát
- Nhấn `s` để dừng cảnh báo thủ công (khi đang bật)
- Sau khi dừng cảnh báo thủ công, chờ một khoảng (cooldown) trước khi cảnh báo có thể kích hoạt lại

---

## 🔔 Kiểu cảnh báo: beep (mặc định) hoặc giọng nói (TTS)
Chương trình mặc định phát âm thanh "beep" ngắn lặp lại. Bạn có thể bật giọng nói hệ thống để nói trực tiếp câu cảnh báo.

1) Sử dụng beep (mặc định)
```bash
python3 drowsiness_detector.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

2) Sử dụng giọng nói (TTS)
```bash
python3 drowsiness_detector.py --shape-predictor shape_predictor_68_face_landmarks.dat --tts
```

3) Tùy chỉnh nội dung giọng nói
```bash
python3 drowsiness_detector.py --shape-predictor shape_predictor_68_face_landmarks.dat --tts --alarm-tts-text "Thức dậy, phát hiện buồn ngủ"
```

Ghi chú hệ điều hành:
- macOS: sử dụng `say` (mặc định có sẵn)
- Linux: cần `spd-say` hoặc `espeak`/`paplay`/`aplay` để TTS hoạt động
- Windows: sử dụng PowerShell System.Speech (thường có sẵn)

---

## 🔊 Tính năng nổi bật
- Phát hiện buồn ngủ theo EAR thời gian thực
- Hệ thống cảnh báo liên tục cho đến khi dừng
- Cảnh báo bằng hình ảnh và âm thanh/giọng nói
- Điều khiển thủ công: nhấn 'S' để dừng
- Tham số dễ điều chỉnh để tinh chỉnh độ nhạy
- Tùy chọn tăng cường hình ảnh cho điều kiện ánh sáng kém
- Chế độ debug để xử lý lỗi detection

---

## 🧩 Nâng cấp trong tương lai
- Thêm phát hiện ngáp (mouth aspect ratio)
- Thử dùng Mediapipe hoặc mô hình học sâu để nâng cao độ chính xác
- Gửi cảnh báo/telemetry về hệ thống ITS trung tâm

---

Tác giả: **Your Name**  
Ngày: October 2025
