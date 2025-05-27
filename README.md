# Chest X-ray Classifier

A simple web app to classify chest X-ray images into three categories: **COVID**, **Pneumonia**, or **Normal**, built using PyTorch and Streamlit.

---

## Features

- Upload chest X-ray images (JPG, PNG, JPEG formats)
- Classify images into COVID, Pneumonia, or Normal using a pretrained ResNet18 model
- Display uploaded image alongside prediction result
- Runs on CPU or GPU (if available)

---

## Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed.

### Install dependencies

```bash
pip install streamlit torch torchvision pillow
