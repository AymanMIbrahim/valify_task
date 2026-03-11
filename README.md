<p align="center">
  <img src="./logo.png" alt="Logo" width="450"/>
</p>

# SpoofFormer Face Anti-Spoofing System

A complete **end-to-end Face Anti-Spoofing pipeline** built with:

* **Vision Transformer (SpoofFormer-style architecture)**
* **PyTorch training pipeline**
* **ONNX model export**
* **ONNX Runtime inference**
* **FastAPI API service**
* **Docker deployment**
* **Structured logging and production-ready API**

The system classifies face images into:

```
LIVE  → real person
SPOOF → presentation attack (photo / screen / replay)
```

---

# System Architecture

```
Training Pipeline (PyTorch)
        │
        │
        ▼
Best Model Checkpoint (.pth)
        │
        │
        ▼
ONNX Export
        │
        ▼
spoofformer_best.onnx
        │
        ▼
FastAPI Inference Service
        │
        ▼
Docker Container
        │
        ▼
REST API /predict
```

---

# Repository Structure

```
├── inference
│   ├── checkpoints
│   │   ├── spoofformer_best.onnx
│   │   └── spoofformer_best.onnx.data
│   ├── helpers
│   │   ├── checkpoint.py
│   │   ├── config.py
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── onnx_session.py
│   │   ├── predict.py
│   │   └── preprocess.py
│   ├── models
│   │   ├── __init__.py
│   │   └── spoofformer.py
│   ├── routes
│   │   ├── __init__.py
│   │   └── predict.py
│   └── main.py
├── train
│   ├── checkpoints
│   │   ├── spoofformer_best.onnx
│   │   ├── spoofformer_best.onnx.data
│   │   ├── spoofformer_best.pth
│   │   └── spoofformer_last.pth
│   ├── dataset
│   │   ├── dataset.py
│   │   └── __init__.py
│   ├── helpers
│   │   ├── checkpoint.py
│   │   ├── config.py
│   │   ├── dataloaders.py
│   │   ├── export.py
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── models
│   │   ├── __init__.py
│   │   └── spoofformer.py
│   └── main.py
├── Dockerfile
├── README.md
└── requirments.txt

```

---

# Dataset Format




First Create a folder called data inside train folder (eg. train/data)
download the dataset using this link: https://drive.google.com/file/d/1CMFidVy1u0FuOJ9RdmbDgY5SLBvdgCV1/view?usp=sharing

The training dataset must follow this format:
```
data/
│
├── image1.png
├── image2.png
├── image3.png
│
├── LIVE_TRAIN.txt
├── LIVE_TEST.txt
├── SPOOF_TRAIN.txt
└── SPOOF_TEST.txt
```

Each text file contains **image filenames**.

Example:

```
LIVE_TRAIN.txt
```

```
img001.png
img002.png
img003.png
```

Example:

```
SPOOF_TRAIN.txt
```

```
attack001.png
attack002.png
attack003.png
```

Labels are mapped internally as:

```
0 → spoof
1 → live
```

---

# Training Pipeline

Training uses **PyTorch** with a **Vision Transformer architecture** inspired by SpoofFormer.


```
                RGB Image
                     │
                     ▼
                Conv Stem
                     │
                     ▼
         Multi-Scale Token Embedding
                     │
                     ▼
           HR-ViT Multi-Branch Network
       ┌────────┬────────┬────────┬────────┐
       │Branch1 │Branch2 │Branch3 │Branch4 │
       │HighRes │MidRes  │LowRes  │VeryLow │
       └────────┴────────┴────────┴────────┘
                     │
                     ▼
               RGB Feature Vector


                Depth Image
                     │
                     ▼
                Conv Stem
                     │
                     ▼
         Multi-Scale Token Embedding
                     │
                     ▼
           HR-ViT Multi-Branch Network
       ┌────────┬────────┬────────┬────────┐
       │Branch1 │Branch2 │Branch3 │Branch4 │
       │HighRes │MidRes  │LowRes  │VeryLow │
       └────────┴────────┴────────┴────────┘
                     │
                     ▼
             Depth Feature Vector


                     │
                     ▼
                Feature Fusion
                     │
                     ▼
              Classification Head
                     │
                     ▼
                 Real / Spoof
```

Input resolution:

```
224 x 224 RGB
```

---

## Training

Run training with:

```
python train/main.py
```

During training the system will:

1. Train the model
2. Evaluate after each epoch
3. Save checkpoints
4. Export the best model to ONNX

---

## Metrics

Training reports:

```
Accuracy
Precision
Recall
F1 score
```

Anti-spoofing specific metrics:

```
APCER → Attack Presentation Classification Error Rate
BPCER → Bona Fide Presentation Classification Error Rate
ACER  → (APCER + BPCER) / 2
```

Confusion matrix is also printed:

```
TP TN FP FN
```
---

# Model Evaluation Results

After training the **SpoofFormer Vision Transformer**, the model was evaluated on the **held-out test dataset** using the evaluation pipeline located in:

```text
train/helpers/evaluate.py
```

The evaluation computes both **standard classification metrics** and **face anti-spoofing specific metrics** commonly used in biometric security systems.

---

## Test Set Performance

| Metric    | Value      |
| --------- |------------|
| Accuracy  | **94.72%** |
| Precision | **73.17%** |
| Recall    | **77.12%** |
| F1 Score  | **75.09%** |

These results indicate that the model performs strongly in overall classification accuracy while maintaining reasonable precision and recall for detecting spoof attacks.

---

## Confusion Matrix

|                  | Predicted Spoof | Predicted Live |
| ---------------- |-----------------|----------------|
| **Actual Spoof** | TN = **3267**   | FP = **110**   |
| **Actual Live**  | FN = **89**     | TP = **300**   |

Interpretation:

* **True Positives (TP)** → correctly detected real faces
* **True Negatives (TN)** → correctly detected spoof attacks
* **False Positives (FP)** → spoof attacks incorrectly classified as real (security risk)
* **False Negatives (FN)** → real users incorrectly classified as spoof (usability issue)

---

## Anti-Spoofing Metrics

Face anti-spoofing systems are typically evaluated using **ISO/IEC biometric presentation attack detection metrics**.

| Metric | Value      |
| ------ |------------|
| APCER  | **3.26%**  |
| BPCER  | **22.88%** |
| ACER   | **13.07%** |

### Metric Definitions

**APCER — Attack Presentation Classification Error Rate**

```text
Spoof images classified as Live
```

Measures the **security risk** of spoof attacks bypassing the system.

Lower APCER means better protection against attacks.

---

**BPCER — Bona Fide Presentation Classification Error Rate**

```text
Real users classified as Spoof
```

Measures **usability impact** (false rejection of legitimate users).

---

**ACER — Average Classification Error Rate**

```text
ACER = (APCER + BPCER) / 2
```

This is the **primary metric used in many anti-spoofing benchmarks**.

---

## Interpretation

Key observations from the evaluation:

* The model achieves **high overall accuracy (94.88%)**.
* **APCER is very low (2.13%)**, meaning the system rarely allows spoof attacks to pass.
* **BPCER is higher (31.11%)**, indicating the model occasionally rejects legitimate users.
* The resulting **ACER of 16.62%** shows balanced performance but suggests that further improvements could reduce false rejections.

---

# Model Export

After training completes:

```
train/helpers/export.py
```

exports the best model:

```
spoofformer_best.pth
      ↓
spoofformer_best.onnx
```

ONNX export uses:

```
torch.onnx.export
opset = 17
dynamic batch dimension
```

---

# Inference Pipeline

Inference runs entirely with:

```
ONNX Runtime
```

No PyTorch is required during prediction.

Pipeline:

```
Input image
     │
Resize → Normalize
     │
ONNX Runtime
     │
Softmax
     │
Prediction
```

Output format:

```json
{
  "predicted_label": "live",
  "confidence": 0.98,
  "probabilities": {
    "spoof": 0.02,
    "live": 0.98
  }
}
```

---

# API Service

The system exposes a REST API using **FastAPI**.

Endpoint:

```
POST /predict
```

Upload a face image and receive classification.

---

## Example Request

```
POST /predict
Content-Type: multipart/form-data
```

Upload:

```
file=image.png
```

---

## Example Response

```json
{
  "filename": "face.png",
  "content_type": "image/png",
  "result": {
    "predicted_index": 1,
    "predicted_label": "live",
    "confidence": 0.9912,
    "probabilities": {
      "spoof": 0.0088,
      "live": 0.9912
    }
  }
}
```

---

# API Documentation

Swagger UI:

```
http://localhost:8000/docs
```

Health check:

```
GET /
```

Response:

```
{
  "message": "API is running"
}
```

---

# Logging

The API uses structured logging.

Example logs:

```
2026-03-10 12:40:22 | INFO | spoofformer_api | Starting SpoofFormer ONNX Inference API...
2026-03-10 12:40:22 | INFO | spoofformer_api | ONNX model loaded successfully.
2026-03-10 12:40:22 | INFO | spoofformer_api | ONNX input name: input
2026-03-10 12:40:22 | INFO | spoofformer_api | ONNX output name: logits
```

Prediction request:

```
Received prediction request | filename=test.png
Prediction completed | predicted_label=live | confidence=0.98
```

Errors are logged automatically.

---

# Docker Deployment and Run

Build the Docker image:

```
docker build -t spoofformer-api .
```

Run the container:

```
docker run --rm -p 8000:8000 spoofformer-api
```

Open the API and Test it:

```
http://localhost:8000/docs
```

---

# Train The Model

Inside ./valify_task/

Create Virtual env (eg. venv):

```
conda create -p venv python==3.11
```

Activate the Virtual Env:

```
conda activate ./venv
```

Install requirments.txt:

```
pip install -r requirments.txt
```

- Create ./valify_task/train/data/
- Extract the downloaded dataset inside this folder
- Feel Free to play with the /valify_task/helpers/config.py it has all model hyper parameters
- Run the main script to start train /valify_task/train/main.py

```
python main.py
```

---

# Dependencies

Main dependencies:

```
torch
torchvision
fastapi
uvicorn
python-multipart
pillow
numpy
scikit-learn
opencv-python-headless
onnx
onnxruntime
onnxscript
email-validator
```

Install locally:

```
pip install -r requirments.txt
```

---

# Production Notes

For production deployment:

Recommended improvements:

* GPU ONNX Runtime (`onnxruntime-gpu`)
* Request rate limiting
* Authentication layer
* Batch inference support
* Prometheus metrics
* Kubernetes deployment

---

# Future Improvements

Possible enhancements:

```
Domain generalization training
Multi-scale SpoofFormer architecture
IR / Depth modality support
Model quantization (INT8 ONNX)
TensorRT inference
Face detection integration
```

---