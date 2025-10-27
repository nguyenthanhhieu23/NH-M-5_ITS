import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

# --- Khởi tạo pyttsx3 để cảnh báo âm thanh ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# --- Mở camera ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Khởi tạo face detector và shape predictor ---
face_detector = dlib.get_frontal_face_detector()
predictor_path = r"D:\Zalo Data\driver_drowsiness_ITS\NH-M-5_ITS\NH-M-5_ITS\shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)

# --- Hàm tính Eye Aspect Ratio (EAR) ---
def detect_eye(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

# --- Ngưỡng EAR ---
EAR_THRESHOLD = 0.25

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Không thể lấy frame từ camera")
        continue

    # Ép kiểu ảnh và chuyển sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # Phát hiện khuôn mặt
    faces = face_detector(gray)

    for face in faces:
        landmarks = face_predictor(gray, face)
        leftEye = []
        rightEye = []

        # Right eye: points 42-47
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1 if n != 47 else 42
            x2 = landmarks.part(next_point).x
            y2 = landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        # Left eye: points 36-41
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1 if n != 41 else 36
            x2 = landmarks.part(next_point).x
            y2 = landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        # Tính EAR trung bình
        ear_left = detect_eye(leftEye)
        ear_right = detect_eye(rightEye)
        ear_avg = round((ear_left + ear_right) / 2.0, 2)

        # Nếu mắt nhắm
        if ear_avg < EAR_THRESHOLD:
            cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
            cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
            engine.say("Alert!!!! WAKE UP DUDE")
            engine.runAndWait()

        # Hiển thị EAR
        cv2.putText(frame, f"EAR: {ear_avg}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Drowsiness Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # nhấn 'q' để thoát
        break

cap.release()
cv2.destroyAllWindows()
