import sys
sys.path.append('./src')
sys.path.append('./src/models')
import torch.nn as nn
import numpy as np
import dataman.wrangle as wrangle
from models.seq2seq_model_pytorch import Seq2SeqPytorch
import models.model_pipeline_pytorch as model_pipeline_pytorch
import models.siamese_pytorch as siamese_pytorch
from utils import dotdict
import torch
import torch.optim as optim
import sys
import logging

import models.load_embeddings as load_embeddings
import constants

args = dotdict({
    'encoder_type': 'transformer',
    'lr': 0.01,
    'learning_rate_decay': 0.9,
    'max_length': 50,
    'epochs': 20,
    'batch_size': 128,
    'batches_per_epoch': 5000,
    'test_batches_per_epoch': 500,
    'input_size': 300,
    'hidden_size': 512,
    'embedding_size': 300,
    'n_layers': 1,
    'bidirectional': False,
    'fix_emb': True,
    'd_proj': None,
    'dp_ratio': 0.7,
    'd_out': 3,  # 3 classes
    'mlp_classif_hidden_size_list': [512, 512],
    'cuda': torch.cuda.is_available(),
})
state = {k: v for k, v in args.items()}


if __name__ == "__main__":
    print(args)

    dm = wrangle.DataManager(args)
    args.n_embed = dm.vocab.n_words
    if True:
        model = siamese_pytorch.SiameseClassifier(config=args)
        model.embed.weight.data = load_embeddings.load_embeddings(
            dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)
        model = dotdict({
            'net': model,
            'criterion': nn.NLLLoss(),
        })  # sorry!
    else:
        model = Seq2SeqPytorch(args=args, vocab=dm.vocab)
        model.net.encoder.embedding.weight.data = load_embeddings.load_embeddings(
            dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)
