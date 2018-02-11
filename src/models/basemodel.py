

#Parent class for all models
class BaseModel:
    def __init__(self):
        pass

    # Train the model
    def train(self, train_data):
        raise NotImplementedError()

    # Calculate the percent correct on an eval set
    def eval(self, eval_data):
        raise NotImplementedError
