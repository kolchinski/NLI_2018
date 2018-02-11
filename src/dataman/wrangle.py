import constants

class DataManager:
    def __init__(self, data):
        self.data = data



def load_train_data():
    with open(constants.TRAIN_DATA_PATH) as f:
        lines = f.readlines()
        key = lines[0]
        data = lines[1:]
        print(key + '\n')
        print(len(lines))
        for i in range(10):
            print(data[i])
