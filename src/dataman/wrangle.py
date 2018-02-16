import constants
import json

class DataManager:
    def __init__(self, train_data, dev_data, test_data):
        self.train_data = train_data
        self.train_x1s = [d['sentence1'] for d in self.train_data]
        self.train_x2s = [d['sentence2'] for d in self.train_data]
        self.train_ys  = [d['gold_label'] for d in self.train_data]

        self.dev_data = dev_data
        self.dev_x1s = [d['sentence1'] for d in self.dev_data]
        self.dev_x2s = [d['sentence2'] for d in self.dev_data]
        self.dev_ys  = [d['gold_label'] for d in self.dev_data]

        self.test_data = test_data
        self.test_x1s = [d['sentence1'] for d in self.test_data]
        self.test_x2s = [d['sentence2'] for d in self.test_data]
        self.test_ys  = [d['gold_label'] for d in self.test_data]


def load_data():
    with open(constants.SMALL_TRAIN_DATA_PATH) as f:
        train_data = [json.loads(l) for l in f.readlines()]
    with open(constants.DEV_DATA_PATH) as f:
        dev_data = [json.loads(l) for l in f.readlines()]
    with open(constants.TEST_DATA_PATH) as f:
        test_data = [json.loads(l) for l in f.readlines()]
    return train_data, dev_data, test_data
