# ðŸ‹ï¸ Squat ML Detection

A computer vision pipeline that analyses squat form from video using **MediaPipe Pose** landmarks. Extracts biomechanical metrics â€” depth, acceleration, ascent/descent time â€” exports them to CSV, and trains a scikit-learn classifier on the collected data.

## ðŸŽ¯ What I Built

A tool for recording and classifying squat repetitions from video. MediaPipe's pose estimation extracts per-frame landmark positions; derived metrics are saved to CSV for model training. A scikit-learn model is then trained on ~100 labelled reps to classify form quality.

## âœ¨ Features

- **Pose landmark extraction** â€” MediaPipe Pose on full-body video
- **Biomechanical metrics** â€” average acceleration, depth, ascent/descent time
- **Multi-video workflow** â€” cycle through a batch of videos sequentially
- **CSV export** â€” save metrics per rep for offline model training
- **Scikit-learn classifier** â€” trained on extracted features to classify squat quality
- **Screenshot capture** â€” save frames mid-analysis

## ðŸ•¹ï¸ Controls

| Key | Action |
|-----|--------|
| `N` | Next video |
| `S` | Screenshot |
| `B` | Save current rep metrics to CSV |

## ðŸ› ï¸ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Pose Estimation | MediaPipe Pose |
| Computer Vision | OpenCV |
| ML Model | scikit-learn |
| Data | NumPy / CSV |

## ðŸš€ Getting Started

```bash
git clone https://github.com/okupacolossal/Squat-ML-Detection
cd Squat-ML-Detection
pip install mediapipe opencv-python scikit-learn numpy
python main.py
```

Place your squat videos in the project folder and navigate through them with `N`.

## ðŸ’¡ What I Learned

- Extracting and interpreting MediaPipe Pose landmarks for biomechanical analysis
- Deriving time-series metrics (velocity, depth, phase timing) from landmark sequences
- Building a data collection pipeline that feeds directly into ML training
- Training a scikit-learn classifier on a small, self-collected dataset (~100 samples)
- Understanding the full ML loop: data capture â†’ feature engineering â†’ model training â†’ inference
