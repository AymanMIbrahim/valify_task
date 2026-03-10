from pathlib import Path

# =========================
# Labels
# =========================
CLASS_NAMES = ["spoof", "live"]
CLASS_TO_IDX = {"spoof": 0, "live": 1}
IDX_TO_CLASS = {0: "spoof", 1: "live"}

# =========================
# Image / Model settings
# =========================
IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 2

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
ONNX_MODEL_PATH = BASE_DIR / "checkpoints" / "spoofformer_best.onnx"