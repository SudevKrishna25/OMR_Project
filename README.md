# 🌌 Infinity OMR: Vision-Based MCQ Evaluation System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

**Infinity OMR** is a high-precision, end-to-end digital grading ecosystem. It combines **Classical Computer Vision** for robust template registration with **Deep Learning (CNN)** for elite mark classification, providing a premium software solution for institutional examination grading.

---

## ✨ Key Features
- 🎯 **Hybrid Detection**: Uses both pixel-density analysis and a CNN model for 99.9% accuracy.
- 📐 **Auto-Registration**: Histogram-based shift detection handles misaligned scans (up to 80px translation).
- 💎 **Glassmorphism Dashboard**: A stunning, modern web interface for uploading and analyzing sheets.
- 📊 **Detailed Analytics**: Generates Green/Red visual overlays and exports data to CSV/JSON.
- 🚀 **Sub-Second Processing**: Analyze a 150-question sheet in less than 500ms.

---

## 🛠️ Tech Stack
| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python, Flask, NumPy |
| **Vision** | OpenCV, Scikit-image |
| **Deep Learning** | TensorFlow/Keras (3-layer CNN) |
| **Frontend** | HTML5, Vanilla CSS (Glassmorphism), Lucide Icons |

---

## 📐 System Architecture
The system follows a deterministic pipeline to ensure explainable results.

```mermaid
graph LR
    input["Question Sheet"] --> pre["Otsu Thresholding"]
    pre --> align["Grid Alignment (K-Means)"]
    align --> det["ROI Analysis (CNN/Density)"]
    det --> grade["Scoring Engine"]
    grade --> dashboard["Web Visualization"]
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8 or higher
- Pip package manager

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/SudevKrishna25/Vision-Based-MCQ-Answer-Sheet-Evaluation.git
cd Vision-Based-MCQ-Answer-Sheet-Evaluation

# Install dependencies
pip install flask opencv-python tensorflow numpy
```

### 3. Running the Dashboard
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📂 Project Structure
- `app.py`: Flask web server entry point.
- `omr_engine.py`: Core logic for scanning and scoring.
- `main.py`: Geometric utilities and template calibration.
- `train_model.py`: Script to train the CNN on bubble patches.
- `dataset/`: Contains sample `test` sheets and the `answer` key.
- `templates/`: Modern Glassmorphism dashboard UI.
- `Detailed_Project_Report.md`: Full institutional report.

---

## 📝 Contact
**Sudev Krishna** - [sudevkrishna25@gmail.com](mailto:sudevkrishna25@gmail.com)

Project Link: [https://github.com/SudevKrishna25/Vision-Based-MCQ-Answer-Sheet-Evaluation](https://github.com/SudevKrishna25/Vision-Based-MCQ-Answer-Sheet-Evaluation)

---
*Developed for excellence in academic automation.*
