


def eval_model(Model, train_data, eval_data):
    model = Model()
    model.train(train_data)
    eval_score = model.eval(eval_data)
    return eval_score