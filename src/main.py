import dataman.wrangle as wrangle
from models.naivemodel import NaiveModel
import eval



print("Hello toxic comments world!")
dm = wrangle.DataManager(*wrangle.load_data())

print('\n')
print(dm.train_x1s[0])
print(dm.train_x2s[0])
print(dm.train_ys[0])


# Evaluate "always neutral" model on train data
print("\nTesting naive model on small training set. Accuracy:")
print(eval.eval_model(NaiveModel, None, dm.train_data))
