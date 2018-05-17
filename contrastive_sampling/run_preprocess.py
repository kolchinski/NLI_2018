# python 3
import json
import pandas as pd
from sample_squad_contrastive import sample_squad_classif_dataset
from sample_nli_contrastive import sample_nli_classif_dataset

print('sampling train allnli')
path = '/Users/yukatherin/Downloads/train_yoda'
with open(path) as f:
    source_data = pd.read_csv(f, sep='\t', header=None, quotechar='"')
sample_nli_classif_dataset(source_data=source_data,
    output_path='/Users/yukatherin/Downloads/train_allnli_classif_k1.txt')


print('sampling train allnli')
path = '/Users/yukatherin/Downloads/tune_yoda'
with open(path) as f:
    source_data = pd.read_csv(f, sep='\t', header=None, quotechar='"')
sample_nli_classif_dataset(source_data=source_data,
    output_path='/Users/yukatherin/Downloads/tune_allnli_classif_k1.txt')


import sys; sys.exit(0)

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
print('sampling train squad')
path = '/Users/yukatherin/Downloads/train-v1.1.json'
with open(path) as f:
    source_data = json.load(f)
_, _ = sample_squad_classif_dataset(gl_vocab=gl_vocab, source_data=source_data,
    output_path='/Users/yukatherin/Downloads/train_squad_classif.txt')

# sample examples - dev
print('sampling dev squad')
path = '/Users/yukatherin/Downloads/dev-v1.1.json'
with open(path) as f:
    source_data = json.load(f)
_, _ = sample_squad_classif_dataset(gl_vocab=gl_vocab, source_data=source_data,
    output_path='/Users/yukatherin/Downloads/dev_squad_classif.txt')
