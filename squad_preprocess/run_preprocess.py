# python 3
import json
from sample_dataset import sample_squad_classif_dataset

# load GloVe for vocab
print('reading glove vectors')
gl_vocab = set()
with open('/Users/yukatherin/Downloads/glove.840B.300d.txt') as f:
    for i, line in enumerate(f.readlines()):
        if i % 100000 == 0:
            print(i)
        gl_vocab.add(line.split()[0])
assert 'happy' in gl_vocab

# sample examples - train
print('sampling train')
path = '/Users/yukatherin/Downloads/train-v1.1.json'
with open(path) as f:
    source_data = json.load(f)
_, _ = sample_squad_classif_dataset(gl_vocab=gl_vocab, source_data=source_data,
    output_path='/Users/yukatherin/Downloads/train_squad_classif.txt')

# sample examples - dev
print('sampling dev')
path = '/Users/yukatherin/Downloads/dev-v1.1.json'
with open(path) as f:
    source_data = json.load(f)
_, _ = sample_squad_classif_dataset(gl_vocab=gl_vocab, source_data=source_data,
    output_path='/Users/yukatherin/Downloads/dev_squad_classif.txt')
