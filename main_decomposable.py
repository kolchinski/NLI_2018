import sys
sys.path.append('./src')
sys.path.append('./src/models')
import torch.nn as nn
import numpy as np
import dataman.wrangle as wrangle
from models.seq2seq_model_pytorch import Seq2SeqPytorch
import models.model_pipeline_pytorch as model_pipeline_pytorch
import models.siamese_pytorch as siamese_pytorch
import models.decomposable_pytorch as decomposable_pytorch
from utils import dotdict
import torch
import torch.optim as optim
import sys
import logging

import models.load_embeddings as load_embeddings
import constants

logger = logging.getLogger(__name__)

args = dotdict({
    'encoder_type': 'decomposable',
    'lr': 0.025,
    'learning_rate_decay': 1,
    'para_init': 0.01, # parameter init Gaussian variance
    'optimizer': 'Adagrad',
    'Adagrad_init': 0.,
    'weight_decay': 5e-5,
    'max_length': 100,
    'epochs': 250,
    'batch_size': 32,
    'batches_per_epoch': 18000,
    'test_batches_per_epoch': 500,
    'hidden_size': 200,
    'embedding_size': 300,
    'fix_emb': True,
    'dp_ratio': 0.2,
    'd_out': 3,  # 3 classes
    'max_norm': 5,
    'cuda': torch.cuda.is_available(),
    'display_interval': 10,
})
state = {k: v for k, v in args.items()}


if __name__ == "__main__":
    print(args)

    dm = wrangle.DataManager(args)
    args.n_embed = dm.vocab.n_words

    model = decomposable_pytorch.SNLIClassifier(config=args)
    model.encoder.embedding.weight.data = load_embeddings.load_embeddings(
        dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)
    model = dotdict({
        'net': model,
        'criterion': nn.NLLLoss(),
    })  # sorry!

    best_dev_acc = 0
    best_train_loss = np.infty

    if args.cuda:
        model.net.cuda()

    if args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad([param for param in model.net.parameters() if param.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
        for group in optimizer.param_groups:
            for p in group['params']:
                optstate = optimizer.state[p]
                optstate['sum'] += args.Adagrad_init
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta([param for param in model.net.parameters() if param.requires_grad],
                                   lr=args.lr)
    else:
        optimizer = optim.SGD(
            [param for param in model.net.parameters() if param.requires_grad],
            lr=state['lr'])

    for epoch in range(args.epochs):
        dm.shuffle_train_data()

        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))
        if epoch % args.display_interval == 0:
            print('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))

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
            state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': dev_acc,
                'best_acc': best_dev_acc,
                'optimizer': optimizer.state_dict()
            }, is_best=dev_acc > best_dev_acc)

        if train_loss > best_train_loss:
            state['lr'] *= args.learning_rate_decay
        else:
            best_train_loss = train_loss
        if train_loss > best_train_loss:
            state['lr'] *= args.learning_rate_decay
        else:
            best_train_loss = train_loss
