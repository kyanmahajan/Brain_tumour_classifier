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

### 2.1 Image Transformation

To ensure that images match the input expected by the trained ResNet-18 model:

1. **Resize:** Image’s smaller side resized to 256 pixels  
2. **Center Crop:** Crop a `224x224` region from the center  
3. **ToTensor:** Convert the PIL image into a PyTorch tensor  
4. **Normalization:** Pixel values normalized using ImageNet mean and standard deviation:

```python
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
