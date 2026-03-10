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
# Training settings
# =========================
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "train" / "data"
CHECKPOINT_DIR = BASE_DIR / "train" /"checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "spoofformer_best.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "spoofformer_last.pth"