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

The training dataset must follow this format:

```
First Create a folder called data inside train folder (eg. train/data)
download the dataset using this link:

Formate the data like this:

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

Main components:

```
Patch Embedding
Transformer Blocks
Class Token
MLP Head
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

# Docker Deployment

Build the Docker image:

```
docker build -t spoofformer-api .
```

Run the container:

```
docker run --rm -p 8000:8000 spoofformer-api
```

Open the API:

```
http://localhost:8000/docs
```

---

# Dependencies

Main dependencies:

```
torch
torchvision
onnx
onnxruntime
onnxscript
fastapi
uvicorn
python-multipart
numpy
pillow
opencv-python-headless
scikit-learn
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