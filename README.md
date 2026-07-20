# 🌿 AI-Powered Plant Disease Detection using YOLO11

An AI-powered plant disease classification system developed using **YOLO11**, **Python**, and **Streamlit**. The application allows users to upload an image of a plant leaf and predicts the corresponding disease with a confidence score while also providing disease-specific recommendations.

---

## Overview

Plant diseases significantly impact agricultural productivity. Early detection helps farmers take timely preventive measures and improve crop yield.

This project utilizes a trained **YOLO11 Classification** model to identify diseases from plant leaf images belonging to multiple crop species.

---

## Features

- Upload plant leaf images through an interactive Streamlit interface.
- Disease prediction using a trained YOLO11 classification model.
- Displays prediction confidence.
- Provides disease-specific recommendations and preventive measures.
- Supports healthy and diseased plant identification.
- Clean and user-friendly interface.

---

## Technologies Used

- Python
- YOLO11 (Ultralytics)
- Streamlit
- PyTorch
- OpenCV
- Pillow (PIL)
- NumPy

---

## Dataset

The model was trained using the **New Plant Diseases Dataset (Augmented)**.

### Dataset Statistics

| Property | Value |
|-----------|-------|
| Dataset Size | 1.4 GB |
| Total Images | 87,867 |
| Plant Species | 14 |
| Disease Categories | 38 |
| Image Format | JPEG |
| Image Resolution | 256 × 256 |
| Image Channels | RGB |

### Supported Plants

- Apple
- Blueberry
- Cherry
- Corn (Maize)
- Grape
- Orange
- Peach
- Bell Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

---

## Project Structure

```
Project_Exhibition/
│
├── Program.py                      # Main Streamlit application
├── Yolo11_Predict.py               # Standalone prediction script
├── Plant_Disease_Classification.pt # Trained YOLO11 model
├── Potato_Disease_Classification.pt
├── best.pt
├── GUI.sh                          # Launch script
├── Image_Info.txt                  # Dataset information
├── requirements.txt
├── README.md
├── .gitignore
└── demo.mp4
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/Plant-Disease-Detection.git
```

Move into the project directory.

```bash
cd Plant-Disease-Detection
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

Run the Streamlit application.

```bash
streamlit run Program.py
```

---

## How It Works

1. Upload a plant leaf image.
2. The image is processed by the trained YOLO11 classification model.
3. The predicted disease class is displayed.
4. Prediction confidence is shown.
5. Disease-specific recommendations are provided for infected plants.

---

## Model

- Architecture: YOLO11 Classification
- Framework: Ultralytics YOLO
- Input Size: 256 × 256 pixels
- Output: Disease Class + Confidence Score

---

## Sample Output

After uploading a leaf image, the application displays:

- Uploaded image
- Predicted disease
- Confidence score
- Recommended treatment or preventive action

---

## Demo

A demonstration of the application is available in **demo.mp4**.

---

## Future Improvements

- Real-time webcam detection
- Mobile application
- Disease severity estimation
- Multi-language support
- Fertilizer recommendation
- Weather-based disease prediction
- Deployment on cloud platforms

---

## Author

Developed as an academic project for plant disease detection using Deep Learning and YOLO11.