from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train"
VAL_DIR = DATA / "val"
TEST_DIR = DATA / "test"

# NO_VAL = True
NO_VAL = False
