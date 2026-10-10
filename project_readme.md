# 🐾 Pet Breed Classifier 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_LINK_HERE)

## Overview
A web application that classifies images of dogs and cats into 37 specific breeds. Built with PyTorch and Streamlit, this project leverages a fine-tuned ResNet-34 model trained on the Oxford-IIIT Pet Dataset. 

## The Problem
Identifying specific pet breeds visually can be challenging due to high inter-class similarity (e.g., different types of terriers or shorthair cats). This project aims to automate breed classification using computer vision, providing a lightweight, accessible web interface for real-time inference.

## My Approach
* **Dataset:** Oxford-IIIT Pet Dataset (37 categories: 25 dog breeds, 12 cat breeds).
* **Architecture:** ResNet-34 (Transfer Learning). The fully connected layer was adapted for 37 classes.
* **Deployment:** Hosted dynamically on Streamlit Community Cloud, with model weights (`.pth`) retrieved securely via the Hugging Face Hub to bypass GitHub file size limits.
* **Preprocessing:** Standard ResNet transformations (224x224 resize, RGB conversion, ImageNet normalization).

## Results & Evaluation
*(Note: Add your actual metrics here once you calculate them)*
* **Headline Accuracy:** XX% on the validation set.
* **Confusion Matrix:** *(Add an image of your confusion matrix here to show which similar-looking breeds get mixed up)*.
* **Failure Modes:** Analyzed the worst-performing examples to understand where the model struggles (e.g., extreme close-ups, noisy backgrounds, or specific breed overlaps).

## Honest Limitations
* **Out-of-Distribution Data:** The model was strictly trained on 37 breeds. When presented with unseen breeds (e.g., Siberian Husky or Golden Retriever), the model is forced to guess the closest visual match. I have implemented a 40% confidence threshold warning to handle this gracefully.
* **Background Interference:** The model occasionally struggles when the subject is too small in the frame or obstructed by complex backgrounds.

## 🚀 Future Work & Roadmap
This project is under active development. Current work-in-progress features include:
* **Combine into a Multi-Task Model:** Transitioning from a simple classifier to a shared ResNet encoder feeding two separate heads—one for classification and one for semantic segmentation.
* **Segmentation Overlay:** Updating the UI to return the predicted breed alongside a segmentation mask outlining the animal in the uploaded photo.
* **Rigorous Metric Tracking:** Implementing per-class Intersection over Union (IoU) to strictly evaluate the upcoming segmentation performance.

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/secrabdou/Customer-Segmentation.git
cd Customer-Segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```