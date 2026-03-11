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
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

# =========================
# SpoofFormerNet model settings
# =========================
STEM_DIM = 32
BRANCH_DIMS = [32, 64, 128, 256]
STAGE_DEPTHS = [1, 1, 1, 1]
NUM_HEADS = [2, 4, 4, 8]
WINDOW_SIZES = [7, 7, 7, 7]
SPARSE_STRIDES = [8, 8, 4, 4]
MLP_RATIO = 2.0
DROPOUT = 0.1
ATTN_DROPOUT = 0.1
USE_DEPTH_STREAM = False
MULTISCALE_KERNELS = [3, 5, 7, 9]

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "train" / "data"
CHECKPOINT_DIR = BASE_DIR / "train" /"checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "spoofformer_best.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "spoofformer_last.pth"