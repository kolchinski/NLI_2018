from models.basemodel import BaseModel
import numpy as np

# Always guess "neutral"
class NaiveModel(BaseModel):
    def train(self, train_data):
        pass

    def predict(self, point):
        return "neutral"

    def eval(self, eval_data):
        return np.mean([p['gold_label'] == 'neutral' for p in eval_data])
