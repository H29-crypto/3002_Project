
# 🔍 3002_Project - Weapon Detection with YOLOv12, ResNet Classifier, and Grad-CAM

This project detects **five types of weapons** in X-ray images using a combination of:
- **YOLOv12** for object detection
- **ResNet18** for secondary classification
- **Grad-CAM** for visual explainability

✅ Developed for the Deep Learning course (AIN3002)

---

## 📂 Dataset

- **Name**: SIXray  
- **Link**: [Roboflow Dataset](https://universe.roboflow.com/ford-ez5dh/sixray-uu8u5)  
- **Classes**:
  - Gun
  - Knife
  - Pliers
  - Scissors
  - Wrench  
- **Custom split**:
  - Train: 497 images
  - Validation: 105 images
  - Test: 105 images

---

## 📁 Project Structure
```
📦3002_Project
┣ 📁Data/
┃ ┣ 📁train/
┃ ┣ 📁val/
┃ ┗ 📁test/
┣ 📁Crops/               # YOLO-detected objects for classifier
┣ 📁results/             # mAP curves, predictions, visualizations
┣ 📄threat_classifier.pt # Trained ResNet18 model
┣ 📄best.pt              # YOLOv12 weights
┣ 📄gradcam_classifier.py
┣ 📄crop_objects.py
┣ 📄train_resnet_classifier.py
┣ 📄data.yaml
┣ 📄yolo12s.yaml
┣ 📄README.md
```

---

## 📦 Required Libraries

Install with:

```bash
pip install -r requirements.txt
```

Main Dependencies:
- torch
- torchvision
- ultralytics
- opencv-python
- pillow
- pytorch-grad-cam
- matplotlib

---

## 🚀 How to Run

### 1. Train YOLOv12 on a custom dataset
```python
from ultralytics import YOLO
model = YOLO("yolo12s.yaml")
model.train(data="data.yaml", epochs=200, imgsz=1024)
```

### 2. Crop images using YOLO predictions
```bash
python crop_objects.py
```

### 3. Train ResNet classifier on cropped images
```bash
python train_resnet_classifier.py
```

### 4. Run Grad-CAM to visualize attention
```bash
python gradcam_classifier.py
```

---

## 📈 Evaluation Metrics (YOLOv12)

| Class    | Precision | Recall | mAP@50 | mAP@50-95 |
| -------- | --------- | ------ | ------ | ---------- |
| Gun      | 0.732     | 0.684  | 0.749  | 0.325      |
| Knife    | 0.343     | 0.360  | 0.324  | 0.156      |
| Pliers   | 0.528     | 0.418  | 0.434  | 0.204      |
| Scissors | 0.445     | 0.267  | 0.257  | 0.119      |
| Wrench   | 0.175     | 0.125  | 0.166  | 0.082      |

---

## 🧠 Project Highlights
- 🔍 Two-Stage Detection: YOLOv12 for detection → ResNet for refinement
- 🧠 Explainability: Grad-CAM highlights what the classifier sees
- 🧪 Strong Augmentations: Mosaic, Mixup, HSV, Flip, etc.
- 💻 Google Colab Compatible: Code tested on Google Colab Pro
- 🔐 Classifier Confidence Thresholding to reject low-confidence results

---

## 📌 Future Work
- Add a webcam or a real-time app
- Replace ResNet18 with EfficientNet or ViT
- Improve dataset balance (add non-weapon samples)
- Deploy via Streamlit or Flask (optional)

---

## 🙌 Acknowledgements
- Ultralytics YOLO
- Roboflow
- PyTorch Grad-CAM
