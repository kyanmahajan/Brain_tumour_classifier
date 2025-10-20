
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, jsonify, request, render_template
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import cv2
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from flask_cors import CORS



def get_edge_for_visualization(pil_img):
  
    w, h = pil_img.size
    scale = 256 / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

   
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))

   
    gray = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = 255 - cv2.Canny(blurred, 30, 80)

    
    edge_rgb = np.stack([edges]*3, axis=-1).astype(np.float32) / 255.0

    return edge_rgb

app = Flask(__name__)
CORS(app)
os.makedirs("static", exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transform setup (same as training)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4) 

model.load_state_dict(torch.load("Brain_Xray_Classification.pth", map_location=device))
model.to(device)
model.eval()


class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# @app.route("/")
# def home():
#     return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).convert("RGB")
        input_tensor = data_transforms(image).unsqueeze(0).to(device)  

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_label = class_names[pred_idx]
            classes = [ClassifierOutputTarget(pred_idx)];
            target_layer = [model.layer4[-1]] 

        cam = GradCAM(model = model, target_layers  = target_layer)

        heatmap  = cam(input_tensor  = input_tensor, targets = classes);
        edge_img = get_edge_for_visualization(image)
        cam_image = show_cam_on_image(edge_img, heatmap[0], use_rgb=True)
        input_img  = np.array(image);
        input_img = input_img.astype(np.float32)/255
        input_img = cv2.resize(input_img, (224,224))
        cam_image_real =  show_cam_on_image(input_img, heatmap[0], use_rgb=True)    
        cam_image_path = os.path.join("static", f"cam_{file.filename}")
        cv2.imwrite(cam_image_path, cv2.cvtColor(cam_image_real, cv2.COLOR_RGB2BGR))
        file={
            "prediction": pred_label,
            "cam_image_url": f"/static/cam_{file.filename}"
        }
        print(file)
        return jsonify(file)

             



        
       
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)



