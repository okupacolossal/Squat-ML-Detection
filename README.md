# 🏋️ Squat ML Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00897B?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Real-time squat analysis using **MediaPipe pose estimation** and a **machine-learning grading model**. The system detects squat repetitions automatically, extracts biomechanical metrics (knee angles, depth, descent/ascent timing, velocity smoothness), and grades each rep using a Linear Regression model trained on a labelled dataset.

---

## 📸 Demo

| Pose skeleton overlay | Live angle readout |
|---|---|
| ![Screenshot 1](pose_screenshot_1775685726.png) | ![Screenshot 2](pose_screenshot_1775685728.png) |

> Screenshots captured from the included example videos via the `S` key shortcut.

---

## ✨ Features

- **Automatic rep detection** — state machine tracks *still → descending → bottom → ascending → complete* transitions
- **Biomechanical metrics per rep**
  - Left & right knee angles (real-time)
  - Squat depth (degrees below standing)
  - Descent & ascent time
  - Velocity smoothness (angular jerk)
- **ML rep grading** — Linear Regression model predicts a quality score for every completed rep
- **Multi-video support** — cycle through a folder of videos with a keypress
- **CSV export** — save all rep metrics for further analysis or model retraining
- **Screenshot capture** — save annotated frames on demand

---

## 🗂️ Project Structure

```
Squat-ML-Detection/
│
├── AngleDetection.py        # Main script — pose detection, state machine, grading
├── data.csv                 # Labelled training dataset (100 reps, 8 features + rank)
├── rep_grading_model.pkl    # Pre-trained Linear Regression model (auto-regenerated)
│
├── VideoExamples/           # Sample squat videos for testing
│   ├── *.mp4
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

```
Video frame
    │
    ▼
MediaPipe Pose  ──►  33 3-D body landmarks
    │
    ▼
Angle Calculation  ──►  hip–knee–ankle angle (left & right)
    │
    ▼
State Machine  ──►  still / descending / ascending / complete
    │                (based on rolling window of last N angles)
    ▼
Rep Metrics  ──►  depth, descent_time, ascent_time, velocity_smoothness
    │
    ▼
ML Grading Model  ──►  quality score  (Linear Regression, scikit-learn)
    │
    ▼
Annotated display via OpenCV
```

### Rep State Machine

The detector maintains a rolling buffer (25 frames) of knee angles and transitions between four states:

| State | Trigger |
|---|---|
| `still` | Angle variance < 10° for ≥ 0.2 s |
| `descending` | Angle drops below rolling average |
| `ascending` | Angle rises above 10-frame average and is ≤ 125° |
| `complete` | Returns to within 5° of start angle |

### ML Model

A **Linear Regression** model is trained at startup from `data.csv`:

| Feature | Description |
|---|---|
| `start_angle` | Knee angle at rep start (°) |
| `bottom_angle` | Knee angle at lowest point (°) |
| `end_angle` | Knee angle when rep completes (°) |
| `descent_time` | Seconds to reach bottom |
| `ascent_time` | Seconds to stand back up |
| `depth` | `180 − bottom_angle` (larger = deeper squat) |
| `velocity_smoothness` | Std-dev of angular velocity (lower = smoother) |

Target: `rank` — a subjective quality score (1–5 scale).  
The model is serialised to `rep_grading_model.pkl` so it can be swapped out without code changes.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A webcam **or** mp4 video files placed in `VideoExamples/`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Squat-ML-Detection.git
cd Squat-ML-Detection

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python AngleDetection.py
```

The script will:
1. Train (or re-train) the grading model from `data.csv`
2. Open the first video in `VideoExamples/`
3. Start displaying the annotated pose window

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `N` | Next video in `VideoExamples/` |
| `P` | Pause / resume |
| `S` | Save annotated screenshot |
| `B` | Export all rep metrics to a timestamped CSV |
| `Q` | Quit |

---

## 📊 Data Format

`data.csv` contains one row per labelled rep:

```
rep_number, start_angle, bottom_angle, end_angle,
descent_time, ascent_time, depth, velocity_smoothness, rank
```

Add your own rows (collected via the `B` export, then manually rated) to improve the model's accuracy.

---

## 🔭 Potential Improvements

- [ ] Live webcam input in addition to video files
- [ ] Hip-angle analysis for torso lean detection
- [ ] Replace Linear Regression with a neural network for non-linear grading
- [ ] GUI dashboard (e.g. Tkinter / Streamlit) showing rep history and score trends
- [ ] Exportable session report (PDF / HTML)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| [MediaPipe](https://mediapipe.dev/) | Real-time 33-point 3-D pose estimation |
| [OpenCV](https://opencv.org/) | Frame capture, drawing, display |
| [NumPy](https://numpy.org/) | Vector maths, angle calculation |
| [SciPy](https://scipy.org/) | Savitzky–Golay smoothing filter |
| [pandas](https://pandas.pydata.org/) | Dataset loading & CSV export |
| [scikit-learn](https://scikit-learn.org/) | Linear Regression model |

---

## 💡 What I Learned

- Extracting and interpreting **MediaPipe Pose landmarks** for biomechanical analysis
- Deriving time-series metrics (velocity, depth, phase timing) from landmark sequences
- Building a **finite-state machine** to robustly segment repetitions in noisy angle data
- Designing a **data-collection pipeline** that feeds directly into ML training
- Training a scikit-learn model on a small, self-collected dataset (~100 samples)
- Understanding the full ML loop: **data capture → feature engineering → model training → inference**

---

## 📄 License

This project is released under the [MIT License](LICENSE).
