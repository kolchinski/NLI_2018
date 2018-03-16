import sys
sys.path.append('./src')
sys.path.append('./src/models')

import SentEval.senteval as senteval
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
import time
from pytorch_classification.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


import models.load_embeddings as load_embeddings
import constants

args = dotdict({
    'type': 'siamese',
    'encoder_type': 'rnn',
    'lr': 0.05,
    'use_dot_attention': True,
    'learning_rate_decay': 0.9,
    'max_length': 100,
    'epochs': 10,
    'batch_size': 64,
    'batches_per_epoch': 3000,
    'test_batches_per_epoch': 500,
    'input_size': 300,
    'hidden_size': 2048,
    'n_layers': 1,
    'bidirectional': False,
    'embedding_size': 300,
    'fix_emb': True,
    'dp_ratio': 0.0,
    'd_out': 3,  # 3 classes
    'mlp_classif_hidden_size_list': [512, 512],
    'cuda': torch.cuda.is_available(),
})
state = {k: v for k, v in args.items()}

if __name__ == "__main__":
    print(args)
    checkpoint = sys.argv[1]
    print('found checkpoint dir {}'.format(checkpoint))

    dm = wrangle.DataManager(args)
    args.n_embed = dm.vocab.n_words
    if True:
        model = siamese_pytorch.SiameseClassifier(config=args)
        model.embed.weight.data = load_embeddings.load_embeddings(
            dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)
    else:
        model = Seq2SeqPytorch(args=args, vocab=dm.vocab)
        model.encoder.embedding.weight.data = load_embeddings.load_embeddings(
            dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)

    model_pipeline_pytorch.load_checkpoint(model, checkpoint=checkpoint)

    model.eval()
    if args.cuda:
        model = model.cuda()
