import constants
import json

class DataManager:
    def __init__(self, data):
        self.full_data = data
        self.x1s = [d['sentence1'] for d in data]
        self.x2s = [d['sentence2'] for d in data]
        self.ys  = [d['gold_label'] for d in data]




def load_train_data():
    with open(constants.SMALL_TRAIN_DATA_PATH) as f:
        data = [json.loads(l) for l in f.readlines()]
        return data
