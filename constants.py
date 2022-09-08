from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train"
VAL_DIR = DATA / "val"
TEST_DIR = DATA / "test"

# TODO
# Never have mutable constants, Oh how I miss C++! lol
# Will get them from CLI by next week
# NO_VAL = True
NO_VAL = False
SUBJECT_WISE = True
# SUBJECT_WISE = False
