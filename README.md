# 👁️ PeepleScan

**PeepleScan** is a real-time people counting and gender detection application powered by **YOLOv8**, **MediaPipe**, and **Streamlit**. It processes live webcam feed or uploaded video to count people, detect their gender, and logs everything with timestamps. The app is ideal for business intelligence, retail analytics, and smart surveillance systems.

---

## 🚀 Features

- 🔍 Real-time detection of people using YOLOv8
- 🙋 Gender classification using a pre-trained deep learning model
- 🧠 FaceMesh integration to improve face detection precision
- 📹 Support for webcam and uploaded video files
- 🧾 Timestamped logs of people count and detected genders
- 📥 One-click CSV download of the session logs
- 🧹 Option to clear logs after session ends

---

## 💼 Use Case Example

PeepleScan can be implemented in **large supermarket stores** at the **main entrance camera**:

- 🧍 Detects the **number of people** entering the store in real time.
- 🚻 Automatically classifies each person by **gender**.
- ⏱️ Logs are stored with **timestamps** for detailed reporting.
- 📈 Retail managers can **correlate gender-specific entry data with sales conversion** rates.
- 🚸 In future versions, **age detection** can be integrated to:
  - Identify **children vs adults**.
  - Track **children-related product interests**.
  - Optimize **inventory and marketing strategy** based on demographic insights.

---

## 📦 Requirements

- Python 3.8+
- OpenCV
- Streamlit
- MediaPipe
- NumPy
- Pandas
- ultralytics (YOLOv8)
- Gender detection model files (`gender_deploy.prototxt`, `gender_net.caffemodel`)

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/peeplescan.git
cd peeplescan
bash```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
bash```

### 3. Download YOLO and Gender Model Files

  -Download yolov8s.pt from Ultralytics

  -Download gender_deploy.prototxt and gender_net.caffemodel from:
    OpenCV's Age-Gender model repo

  -Place the downloaded files in the project root directory.

###🧪 Run the App
```bash
streamlit run app.py
bash```

###📊 Sample Log (CSV Format)
```bash
timestamp,people_count,genders
2025-05-29 14:23:12,2,Male, Female
2025-05-29 14:23:18,1,Female
bash```

###📂 Project Structure

```bash
peeplescan/
├── app.py                     # Main Streamlit app
├── gender_deploy.prototxt     # Gender classification model config
├── gender_net.caffemodel      # Gender classification model weights
├── yolov8s.pt                 # YOLOv8 model weights
├── README.md
└── requirements.txt
bash```
