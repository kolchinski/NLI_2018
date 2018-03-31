import os

ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR =  os.path.join(ROOT_PATH, 'static/snli_1.0')
SMALL_TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'snli_1.0_train_small.jsonl')
FULL_TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'snli_1.0_train.jsonl')
DEV_DATA_PATH = os.path.join(DATA_DIR, 'snli_1.0_dev.jsonl')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'snli_1.0_test.jsonl')
SMALL_TRAIN_TOK_DATA_PATH = os.path.join(DATA_DIR, 'train_small.')
FULL_TRAIN_TOK_DATA_PATH  = os.path.join(DATA_DIR, 'train.')
DEV_TOK_DATA_PATH = os.path.join(DATA_DIR, 'dev.')
TEST_TOK_DATA_PATH  = os.path.join(DATA_DIR, 'test.')
EMBED_DATA_PATH = os.path.join(ROOT_PATH, 'static/glove')

MULTINLI_DIR = os.path.join(ROOT_PATH, 'static/MultiNLI')
MULTINLI_DEV_TOK_DATA_PATH = os.path.join(MULTINLI_DIR, 'dev.')
MULTINLI_TEST_TOK_DATA_PATH = os.path.join(MULTINLI_DIR, 'test.')

SQUAD_DIR = os.path.join(ROOT_PATH, 'static/squad')
SQUAD_FULL_TRAIN_DATA_PATH = os.path.join(SQUAD_DIR, 'train_squad_classif.txt')
SQUAD_DEV_DATA_PATH = os.path.join(SQUAD_DIR, 'dev_squad_classif.txt')
