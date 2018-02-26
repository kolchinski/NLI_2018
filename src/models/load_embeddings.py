import torch
import torch.nn as nn

from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

import io
import os
import logging
import array
from tqdm import tqdm
logger = logging.getLogger(__name__)

PAD_token = 0
EOS_token = 1
UNK_token = 2

def load_embeddings(vocab, path, d_embed, unk_init=torch.Tensor.zero_):
    name = 'glove.6B.' + str(d_embed) + 'd.txt'
    name_pt = name + '.pt'
    path_pt = os.path.join(path,name_pt)

    # Build pretrained embedding vectors
    if os.path.isfile(path_pt):  # Load .pt file if there is any cached
        print('Loading vectors from {}'.format(path_pt))
        itos, stoi, vectors, dim = torch.load(path_pt)
    else:  # Read from Glove .txt file
        path = os.path.join(path,name)
        if not os.path.isfile(path):
            raise RuntimeError('No files found at {}'.format(path))
        try:
            with io.open(path, encoding="utf8") as f:
                lines = [line for line in f]
        except:
            raise RuntimeError('Could not read {} as format UTF8'.format(path))
        print("Loading vectors from {}".format(path))

        itos, vectors, dim = [], array.array(str('d')), None

        for line in tqdm(lines, total=len(lines)):
            entries = line.rstrip().split(" ")
            word, entries = entries[0], entries[1:]
            if dim is None and len(entries) > 1:
                dim = len(entries)
            elif len(entries) == 1:
                logger.warning("Skipping token {} with 1-dimensional "
                               "vector {}; likely a header".format(word, entries))
                continue
            elif dim != len(entries):
                raise RuntimeError(
                    "Vector for token {} has {} dimensions, but previously "
                    "read vectors have {} dimensions. All vectors must have "
                    "the same number of dimensions.".format(word, len(entries), dim))
            vectors.extend(float(x) for x in entries)
            itos.append(word)
        stoi = {word: i for i, word in enumerate(itos)}
        vectors = torch.Tensor(vectors).view(-1, dim)
        print(vectors.shape)
        print('Saving vectors to {}'.format(path_pt))
        torch.save((itos, stoi, vectors, dim), path_pt)
    # Look up vectors for words in vocab in the pretrained vectors
    vocab_vectors = torch.Tensor(vocab.n_words, dim).zero_()
    #print(vocab.index2word)
    for _, (i, token) in enumerate(vocab.index2word.items()):
        if i < 3:  # Skip the first 3 words PAD EOS UNK
            continue
        #token = token.strip(''',.:;"()'/?<>[]{}\|!@#$%^&*''')
        #print(i,token,(token in stoi))
        if token in stoi:
            vocab_vectors[i][:] = vectors[stoi[token]]
        else:
            vocab_vectors[i][:] = unk_init(torch.Tensor(1, dim))
    return vocab_vectors