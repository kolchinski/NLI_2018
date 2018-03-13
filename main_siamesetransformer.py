import sys
sys.path.append('./src')
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

logger = logging.getLogger(__name__)

args = dotdict({
    'encoder_type': 'transformer',
    'lr': 0.05,
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
    'd_proj': None,
    'dp_ratio': 0.0,
    'd_out': 3,  # 3 classes
    'max_norm': 5,
    'mlp_classif_hidden_size_list': [512, 512],
    'cuda': torch.cuda.is_available(),
})
state = {k: v for k, v in args.items()}


if __name__ == "__main__":
    print(args)

    dm = wrangle.DataManager(args)
    args.n_embed = dm.vocab.n_words
    if True:
        model = siamese_pytorch.SNLIClassifier(config=args)
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

    best_dev_acc = 0
    best_train_loss = np.infty

    for epoch in range(args.epochs):
        dm.shuffle_train_data()

        optimizer = optim.SGD(
            [param for param in model.net.parameters() if param.requires_grad],
            lr=state['lr'])
        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))

        if args.cuda:
            model.net.cuda()
        train_loss, train_acc = model_pipeline_pytorch.train(
            model=model.net,
            optimizer=optimizer,
            epoch=epoch,
            di=dm,
            args=args,
            loss_criterion=model.criterion,
        )
        dev_loss, dev_acc = model_pipeline_pytorch.test(
            model=model.net,
            epoch=epoch,
            di=dm,
            args=args,
            loss_criterion=model.criterion,
        )
        if dev_acc > best_dev_acc:
            print('New best model: {} vs {}'.format(dev_acc, best_dev_acc))
            best_dev_acc = dev_acc
            print('Saving to checkpoint')
            model_pipeline_pytorch.save_checkpoint(
                state=state, epoch=epoch, dev_acc=dev_acc)
        if train_loss > best_train_loss:
            state['lr'] *= args.learning_rate_decay
        else:
            best_train_loss = train_loss
