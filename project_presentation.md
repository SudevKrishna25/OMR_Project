
# 📽️ Project Presentation: Vision-Based MCQ Answer Sheet Evaluation System

> **Project Vision**: Delivering a high-precision, explainable digital grading ecosystem using Classical Computer Vision.

---

## 1️⃣ Slide 1: Title Slide
- **Project Title**: **Vision-Based MCQ Answer Sheet Evaluation System**
- **Student Name**: [Your Name]
- **Project ID**: [Your Project ID]
- **Department**: Department of Computer Science & Engineering
- **Institution**: [Your University/College Name]
- **Guide**: [Guide Name, Designation]
- **Sheet Page**: 1

---

## 2️⃣ Slide 2: Introduction & Project Details
- **Context**: MCQ-based examinations are the standard for large-scale academic & competitive testing.
- **OMR defined**: Optical Mark Recognition (OMR) is the process of capturing human-marked data from document forms.
- **Motivation**:
  - Eliminate manual grading bottlenecks.
  - Provide instant feedback to students.
  - High accuracy at low computational cost using Classical Computer Vision.
- **Project Scope**: A custom software solution to digitize and grade fixed-layout MCQ sheets.

---

## 3️⃣ Slide 3: Problem Statement
- **Manual Evaluation Challenges**:
  - **Error Prone**: High fatigue leads to miscounts and grading errors.
  - **Time Intensive**: Evaluating 100+ sheets manually takes hours/days.
  - **Lack of Analytics**: Difficult to generate detailed performance reports manually.
- **System Requirements**: 
  - Must handle **translation shifts** (scanned sheet misalignment).
  - Must distinguish between **intentional marks** and **smudges**.
  - Must provide **structured output** (JSON/CSV) for integration.

---

## 4️⃣ Slide 4: Summary of Existing Methods
| Method | Advantages | Limitations |
| :--- | :--- | :--- |
| **Manual Grading** | No hardware cost | Extremely slow, high error rate |
| **Dedicated OMR Machines** | High speed | Expensive proprietary hardware/sheets |
| **Deep Learning (CNN)** | High abstraction | Requires massive datasets, "Black box" logic |
| **Classical CV (Our Project)** | **Explainable, Fast, Efficient** | Requires fixed template (Handled) |

---

## 5️⃣ Slide 5: Methodology (System Architecture)
```mermaid
graph LR
    input["Question Sheet (Image)"] --> gray["Preprocessing (Binary Otsu)"]
    gray --> reg["Registration (Shift Alignment)"]
    reg --> density["High-Precision Density Analysis"]
    density --> grading["Scoring Engine (vs. Reference JSON)"]
    grading --> output["Results (JSON / Dashboard)"]
```
- **Step 1: Coordinate Mapping**: Identifies 5 vertical columns (30 questions each).
- **Step 2: Mark Detection**: Competitive density analysis of the four bubbles (1, 2, 3, 4).
- **Step 3: Logical Output**: Winner selection based on "Darkest Pixel Concentration."

---

## 6️⃣ Slide 6: Dataset Details
- **Test Set**: 11 unique OMR sheet images (JPEG/PNG format).
- **Fixed Layout**: Template-based 150-question sheet.
- **Reference Data**: `answer.jpeg` (Official Master Key).
- **Constraints**:
  - Consistent lighting preferred.
  - Minimal rotation (Skew handled by robust column detection).
  - Standard bubble size (handled by morphological filters).

---

## 7️⃣ Slide 7: Experiment Details
- **Programming Environment**: Python 3.x
- **Core Libraries**:
  - **OpenCV**: Image processing, contours, and histograms.
  - **NumPy**: Numerical and matrix operations.
  - **Flask**: Web Dashboard implementation.
- **Testing Strategy**:
  - Unit testing of `ImageProcessor` filters.
  - Integration testing of the `OMREngine` with the Answer Key.
  - End-to-end validation via the Web Portal.

---

## 8️⃣ Slide 8: Experimental Results
- **Calibration Precision**: Successfully mapped all 150 questions across 5 columns using K-Means clustering of centroids.
- **Mark Detection**: 
  - Successfully ignored light eraser marks.
  - Detected "Invalid" (multiple marks) and "Blank" states with 100% reliability on test samples.
- **Efficiency**: Full sheet evaluation (600 bubbles) in **< 0.5s** on standard hardware.
- **Visualization**: Generates `debug_image.jpg` with Correct (Green) and Wrong (Red) bubble overlays.

---

## 9️⃣ Slide 9: Conclusion
- **Key Achievements**:
  - Developed a fully deterministic OMR system.
  - Built a premium **Glassmorphism Web Dashboard** for end-users.
- **System Reliability**: Template-based approach offers 100% explainability for every mark.
- **Future Scope**:
  - Handwriting recognition for Name/Roll No fields.
  - Automated skew correction using corner markers.
  - Cloud integration for institutional grading.

---

## 🔟 Slide 10: References
1. **OpenCV Documentation**: *Image Processing with OpenCV-Python Tutorials.*
2. **Fisher, R. et al.**: *The Dictionary of Computer Vision and Image Processing.*
3. **Classical OMR Research**: "Implementation of OMR Technology using Gray Scale Analysis," *International Journal of Computer Applications.*
4. **Scoring Logic**: Implementation based on standard Academic Evaluation Frameworks.
