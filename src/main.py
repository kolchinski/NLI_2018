import dataman.wrangle as wrangle



print("Hello toxic comments world!")
dm = wrangle.DataManager(wrangle.load_train_data())

print('\n\n')
print(dm.x1s[0])
print(dm.x2s[0])
print(dm.ys[0])
