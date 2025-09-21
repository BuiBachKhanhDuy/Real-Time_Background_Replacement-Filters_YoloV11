# 🎭 Real-Time Background Replacement & Filters  

This project is a **computer vision application** that performs:  
- 🔹 Real-time **background replacement** using **YOLOv11 segmentation**  
- 🔹 Fun **face filters** with MediaPipe’s Face Mesh  
- 🔹 Multi-person and single-person handling  
- 🔹 Live webcam streaming with output recording  

---

## 🚀 Features
- Replace webcam background with custom images.  
- Apply overlays/filters (e.g., cat ears) on detected faces.  
- Toggle modes in real time with keyboard shortcuts.  
- Supports multiple people simultaneously.  
- Exports a side-by-side comparison video (`output.avi`).  

---

## 🛠️ Tech Stack
- **Python 3.x**  
- [OpenCV](https://opencv.org/) – image & video processing  
- [YOLOv11 (Ultralytics)](https://github.com/ultralytics/ultralytics) – segmentation model  
- [cvzone](https://github.com/cvzone/cvzone) – easy MediaPipe utilities  
- [MediaPipe](https://developers.google.com/mediapipe) – face mesh detection  
- **NumPy**, **Torch**  

Dependencies are listed in [`requirements.txt`](requirements.txt).  

---

## 📂 Project Structure
```plaintext
├── main.py              # Entry point – runs webcam app
├── bg_utils.py          # Helper for background replacement
├── filters_util.py      # PNG overlay & filter utilities
├── yolo_utils.py        # YOLO segmentation mask extraction
├── requirements.txt     # Dependencies
├── images/              # Background images (user-provided)
├── filters/             # Filter PNGs (e.g., cat ears, glasses)
└── .gitignore           # Ignore env/, __pycache__/, yolov11/

---

## 🎮 Controls
- `1` → Previous background  
- `2` → Next background  
- `m` → Toggle **multi/single person mode**  
- `b` → Toggle **background on/off**  
- `f` → Toggle **filter on/off**  
- `x` → Exit  

---

## ▶️ Run the Project
1. Clone this repository:
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
2. Install dependencies:
  pip install -r requirements.txt
  Download YOLOv11 segmentation weights (e.g., yolo11n-seg.pt) and place inside yolov11/ directory.
3. Run:
  python main.py
