# 🐄 Animal Disease Monitoring System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask" />
  <img src="https://img.shields.io/badge/YOLO-Object%20Detection-purple" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
  <img src="https://img.shields.io/badge/License-Custom-lightgrey" />
</p>

<p align="center">
  **AI-Powered Livestock Health Monitoring & Disease Detection**
</p>

Animal Disease Monitoring System is an intelligent animal disease monitoring system built with **Python, Flask, YOLO, and OpenCV**. It analyzes uploaded cow videos, detects individual cows and visible lesions, and automatically classifies each cow as **Healthy** or **Lumpy** based on lesion detection.

The application provides a real-time annotated video stream, live detection statistics, and automatic report generation — all through a modern browser-based interface.

No manual exports. No download buttons. Just upload, detect, monitor, and automatically save everything.

---

## 📸 Project Overview

AnimalGuard transforms ordinary livestock videos into actionable monitoring information using computer vision and deep learning.

### ✨ Key Highlights

- 🐄 **Cow Detection** — Automatically identifies cows in uploaded videos.
- 🦠 **Lesion Detection** — Detects visible lesions using a YOLO model.
- 🩺 **Health Classification** — Classifies each detected cow as Healthy or Lumpy.
- 🎥 **Live Video Streaming** — Displays annotated video frames directly in the browser.
- 📊 **Live Statistics** — Tracks total cows, healthy cows, and lumpy cows.
- 💾 **Automatic Result Saving** — Saves annotated videos, JSON summaries, and CSV reports.
- 🌐 **LAN Accessibility** — Access the application from multiple devices on the same network.

---

## 🚀 Features

| Feature | Description |
|---|---|
| Video Upload | Upload MP4, AVI, or MOV videos |
| Cow Detection | YOLO-based cow identification |
| Lesion Detection | Detects visible lesions in video frames |
| Health Classification | Healthy vs. Lumpy classification |
| Live MJPEG Stream | Real-time annotated video display |
| Detection Counters | Total Cows, Healthy, and Lumpy |
| Automatic Video Saving | Saves annotated video automatically |
| JSON Reporting | Generates detailed detection summaries |
| CSV Reporting | Stores frame-by-frame detection statistics |
| Responsive Interface | Browser-based monitoring dashboard |
| LAN Support | Access from other devices on the network |

### 📊 Real-Time Monitoring

The dashboard provides:

- Total detected cows
- Number of healthy cows
- Number of lumpy cows
- Individual cow detection details
- Bounding boxes and lesion markers
- Live annotated video feed
- Processing and completion status

### 💾 Automatic Report Generation

After every completed detection session, AnimalGuard automatically generates:

1. 🎥 Annotated video (`.mp4` or `.avi`)
2. 📄 Detection summary (`.json`)
3. 📊 Frame-by-frame report (`.csv`)

All generated files are stored inside the `outputs/` directory.

**No download buttons or manual exports are required.**

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Backend web framework |
| Ultralytics YOLO | Object and lesion detection |
| OpenCV | Video processing and annotation |
| Supervision | Detection visualization utilities |
| NumPy | Numerical operations |
| HTML5 | Web interface |
| CSS3 | Dashboard styling |
| JavaScript | Frontend interactions |
| JSON | Detection summary storage |
| CSV | Frame-level report generation |

---

## 📂 Project Structure

```text
Animal Disease Monitoring System/
│
├── app.py                      # Flask server and detection worker
├── train.py                    # Optional model training script
├── README.md                   # Project documentation
│
├── models/
│   └── yolo11n.pt              # YOLO model weights
│
├── static/
│   ├── css/
│   │   └── style.css           # Dashboard styles
│   │
│   ├── js/
│   │   └── main.js             # Frontend functionality
│   │
│   └── uploads/                # Uploaded source videos
│
├── templates/
│   └── index.html              # Main dashboard
│
└── outputs/                    # Automatically generated results
    ├── *_annotated.mp4
    ├── *_summary.json
    └── *_frames.csv
```

> **Note:** The `static/uploads/` and `outputs/` directories are automatically created when needed.

---

## 🔧 Configuration

Open `Animal Skin Disease Monitoring System.py` and locate the configuration section at the top.

Update the model path according to your computer:

```python
LESION_MODEL_PATH = r"C:\Users\ZaBir\Desktop\Animal Disease Monitoring System\models\yolo11n.pt"

SAVE_ANNOTATED_VIDEO = True

DEFAULT_FPS = 20.0
```

### Configuration Parameters

| Parameter | Description |
|---|---|
| `LESION_MODEL_PATH` | Absolute path to the lesion detection model |
| `SAVE_ANNOTATED_VIDEO` | Enables or disables annotated video saving |
| `DEFAULT_FPS` | Fallback FPS when source video FPS is unavailable |

**Important:** Ensure the specified model file exists before starting lesion detection.

---

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

If everything is configured correctly, the terminal will display messages similar to:

```text
[MODEL] Loading cow detector: ...yolo11n.pt
[SAVE] Outputs will be written to: .../outputs
[START] http://0.0.0.0:5000
```

### Access the Dashboard

Open your browser and visit:

**Local Access**

```text
http://localhost:5000
```

**Network Access**

```text
http://YOUR-LAN-IP:5000
```

You can access the application from another device connected to the same local network.

---

## 🐄 How to Use AnimalGuard

Follow these simple steps:

**Step 1 — Upload Video**

Open the dashboard and upload a livestock video using the file picker or drag-and-drop area.

Supported formats:

- `.mp4`
- `.avi`
- `.mov`

**Step 2 — Start Detection**

Click the **Start Detection** button to begin video processing.

**Step 3 — Monitor Live Detection**

Watch the annotated video feed with:

- Cow bounding boxes
- Individual cow health labels
- Lesion detection markers
- Live detection counters

**Step 4 — View Detection Statistics**

Monitor the real-time counters:

| Counter | Description |
|---|---|
| Total Cows | Total detected cows in the current frame |
| Healthy | Cows classified as healthy |
| Lumpy | Cows classified as lumpy |

**Step 5 — Automatic Saving**

Once video processing finishes:

- The status badge changes to **Saved**.
- Detection counters display the final tally.
- The Auto-Saved Results card displays generated filenames.
- All three output files are stored inside `outputs/`.

> **No manual export is needed.** Open the `outputs/` directory on the server machine to access the generated files.

---

### 3. 📊 CSV Report

The CSV file contains one row per processed video frame.

| Column | Description |
|---|---|
| `frame` | Frame number |
| `cows` | Total detected cows |
| `healthy` | Healthy cow count |
| `lumpy` | Lumpy cow count |
| `frame_total_lesions` | Total detected lesions |

The CSV report is compatible with Microsoft Excel, Google Sheets, and Python Pandas.

---

## 🔌 HTTP API Documentation

AnimalGuard provides the following HTTP endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Loads the web dashboard |
| POST | `/upload` | Uploads a video for detection |
| GET | `/video_feed` | Streams annotated MJPEG frames |
| GET | `/detection_counts` | Returns live detection statistics |
| GET | `/status` | Returns model information and latest output details |

---

## 🧠 How the Detection System Works

AnimalGuard combines object detection and geometric analysis to determine the health status of detected cows.

### Detection Pipeline

```text
        ┌──────────────────────┐
        │    Upload Video      │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Extract Video      │
        │      Frames          │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   YOLO Cow Detection │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   YOLO Lesion        │
        │     Detection        │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Lesion-Cow Overlap │
        │      Analysis        │
        └──────────┬───────────┘
                   │
             ┌─────┴─────┐
             │           │
             ▼           ▼
       ┌──────────┐ ┌──────────┐
       │ Healthy  │ │  Lumpy   │
       └──────────┘ └──────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Annotated Video &    │
        │ Automatic Reports    │
        └──────────────────────┘
```

### 1. Cow Detection

The cow detection model processes every video frame.

Only detections whose class names are `cow` or `cattle` are retained.

### 2. Lesion Detection

The lesion detection model analyzes the same video frame to identify visible lesions.

### 3. Health Classification

The system uses a geometric rule to determine each cow's classification.

- **Lumpy:** At least one detected lesion center point falls inside the cow's bounding box.
- **Healthy:** No detected lesion center point falls inside the cow's bounding box.

### 4. Video Annotation

Each processed frame is annotated with bounding boxes, health labels, and lesion markers.

### 5. Automatic Report Generation

After processing finishes, the system automatically saves the annotated video, JSON summary, and CSV report.

> **Important:** This is a simple geometric classification heuristic, not a clinically validated disease diagnosis. Detection quality depends on the training data, model accuracy, video quality, and confidence thresholds.

---

## 🛠️ Troubleshooting

### ❌ Annotated Video Is Missing

Check the terminal for:

```text
[SAVE] VideoWriter opened with mp4v: ...
```

If you see:

```text
[WARN] No working video codec — annotated video will NOT be saved.
```

Try installing:

```bash
pip install opencv-contrib-python
```

The application attempts the following codecs:

1. `mp4v`
2. `avc1`
3. `XVID` (AVI fallback)

### ❌ Results Card Does Not Appear

- Ensure video processing has completed.
- Check whether the status badge displays **Saved**.
- Check the browser console for JavaScript errors.
- Verify the `/status` endpoint returns populated `last_output` information.

You can test it using:

```bash
curl http://localhost:5000/status
```

### ❌ Model File Not Found

If `LESION_MODEL_PATH` points to a nonexistent file:

- Cow detection can continue if its model loads successfully.
- Lesion detection is disabled.
- Cows without lesion detections will be classified as Healthy.

Place your model weights inside `models/` and update the configuration path.

### ❌ Video Stream Does Not Appear

- Confirm that video upload completed successfully.
- Check that `/upload` returns HTTP 200.
- Try another browser.
- Ensure the browser supports MJPEG streaming using `multipart/x-mixed-replace`.

### ❌ Incorrect Cow Detection Counts

Adjust the confidence thresholds inside `process_video_worker`.

**Cow detection:**

```python
cow_model(frame, conf=0.35)
```

Increase the confidence threshold to reduce false-positive detections.

**Lesion detection:**

```python
lesion_model(frame, conf=0.30)
```

Decrease the confidence threshold to detect more subtle lesions, while monitoring false positives.

### ❌ Port 5000 Already in Use

Change the Flask application port:

```python
app.run(
    host="0.0.0.0",
    port=5001,
    debug=False,
    threaded=True
)
```

Access the application through:

```text
http://localhost:5001
```

---

## ⚠️ Important Disclaimer

AnimalGuard is an educational and research-oriented computer vision project.

Its Healthy/Lumpy classification is based on detected visual lesions and a geometric overlap heuristic. It does not establish a confirmed diagnosis of Lumpy Skin Disease or any other animal disease.

For actual livestock health decisions, consult a qualified veterinarian and use clinically validated diagnostic methods.

---

## 🔮 Future Improvements

Potential future enhancements include:

- 🐄 Individual cow tracking across video frames.
- 🦠 More accurate lesion classification.
- 📈 Historical animal health analytics.
- 📊 Advanced monitoring dashboards.
- 🔔 Automated disease-risk alerts.
- 🗃️ Database integration for detection history.
- ☁️ Cloud-based livestock monitoring.
- 📱 Mobile-friendly monitoring interface.

---

## 👨‍💻 Author

**ZaBir HaSan**

**Year:** 2026

---

<p align="center">
  **🐄 Animal Skin Disease Monitoring System**
  <br/>
  Intelligent Livestock Monitoring with Computer Vision
  <br/><br/>
  <i>Detect • Monitor • Analyze • Protect</i>
</p>
