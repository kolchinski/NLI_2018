import dataman.wrangle as wrangle



print("Hello toxic comments world!")
dm = wrangle.DataManager(*wrangle.load_data())

print('\n\n')
print(dm.train_x1s[0])
print(dm.train_x2s[0])
print(dm.train_ys[0])
