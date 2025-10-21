# Brain X-Ray Classification with Grad-CAM Visualization

## Overview

This project implements a **brain tumor classification system** using a **ResNet-18** convolutional neural network.  
It classifies X-ray images into four categories:

- Glioma  
- Meningioma  
- No Tumor  
- Pituitary  

Additionally, it generates **Grad-CAM visualizations** to highlight regions of the X-ray that influenced the model’s prediction.  
The system is exposed as a **Flask API**, enabling image upload and real-time prediction.

---

## 1. Libraries and Tools Used

- **PyTorch & torchvision** – For model building, transformations, and inference  
- **Flask & Flask-CORS** – To create a web API for uploading and predicting images  
- **PIL & OpenCV** – For image preprocessing and manipulation  
- **pytorch-grad-cam** – For Grad-CAM heatmap generation  
- **NumPy & Matplotlib** – For numerical operations and visualization  

---

## 2. Data Preprocessing Pipeline


To ensure that images match the input expected by the trained ResNet-18 model:

1. **Resize:** Image’s smaller side resized to 256 pixels  
2. **Center Crop:** Crop a `224x224` region from the center  
3. **ToTensor:** Convert the PIL image into a PyTorch tensor  
4. **Normalization:** Pixel values normalized using ImageNet mean and standard deviation:

## Models
Use resnet18 (light weight 18 version)
Train accuracy reacher around 90 while test remained close to 70%
Also tried vision transformers, which performed worse than renset.


