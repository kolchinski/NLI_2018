import os

ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR =  os.path.join(ROOT_PATH, 'static/snli_1.0')
SMALL_TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'snli_1.0_train_small.jsonl')
FULL_TRAIN_DATA_PATH  = os.path.join(DATA_DIR, 'snli_1.0_train.jsonl')