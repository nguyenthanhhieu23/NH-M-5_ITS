# Driver Drowsiness Detection System (ITS Project)

## 🧠 Overview
This project detects driver drowsiness in real-time using a webcam.  
It relies on **dlib's 68 facial landmarks** to calculate the Eye Aspect Ratio (EAR) and trigger an alarm if eyes remain closed for a number of consecutive frames.

---

## ⚙️ Requirements
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

If `dlib` fails to install, try:
```bash
pip install cmake
pip install dlib
```
or use Anaconda:
```bash
conda install -c conda-forge dlib
```

---

## 📦 Files
```
drowsiness_detector.py
requirements.txt
README.md
```

⚠️ You also need the file:
**shape_predictor_68_face_landmarks.dat**
Download from:
https://github.com/davisking/dlib-models  
(Place it in the same folder as the script)

---

## ▶️ Run
```bash
python drowsiness_detector.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

### Options:
| Argument | Description | Default |
|-----------|--------------|----------|
| `--camera` | Camera index | 0 |
| `--ear-thresh` | EAR threshold for closed eyes | 0.25 |
| `--ear-consec-frames` | Frames required to trigger alarm | 20 |
| `--output` | Optional output video file | None |

Press `q` to quit.

---

## 🔊 Features
- Detects eyes and calculates EAR.
- Issues visual + audio alerts when drowsiness is detected.
- Adjustable thresholds and parameters.
- Lightweight and easy to integrate with ITS platforms.

---

## 🧩 Future Work
- Add mouth aspect ratio for yawning detection.
- Use Mediapipe or Deep Learning for higher accuracy.
- Send warning data to central ITS dashboard.

---

Author: **Your Name**  
Date: October 2025
