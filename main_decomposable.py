import sys
sys.path.append('./src')
sys.path.append('./src/models')
import torch.nn as nn
import numpy as np
import dataman.wrangle as wrangle
import models.model_pipeline_pytorch as model_pipeline_pytorch
import models.decomposable_pytorch as decomposable_pytorch
from utils import dotdict
from collections import defaultdict
import torch
import torch.optim as optim
import sys
import logging

import models.load_embeddings as load_embeddings
import constants

logger = logging.getLogger(__name__)

args = dotdict({
    'encoder_type': 'decomposable',
    'intra_attn': True,
    'lr': 0.025,
    'lr_intra': 0.025,
    'learning_rate_decay': 0.9,
    'para_init': 0.01, # parameter init Gaussian variance
    'optimizer': 'Adagrad',
    'Adagrad_init': 0.,
    'weight_decay': 5e-5,
    'max_length': 100,
    'epochs': 50,
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
    if args.intra_attn:
        state['lr'] = args.lr_intra

    dm = wrangle.DataManager(args)
    args.n_embed = dm.vocab.n_words

    model = decomposable_pytorch.SNLIClassifier(config=args)
    model.encoder.embedding.weight.data = load_embeddings.load_embeddings(
        dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)
    criterion = nn.NLLLoss()

    # Numbers of parameters
    print("number of trainable parameters found {}".format(sum(
        param.nelement() for param in model.parameters()
        if param.requires_grad)))

    best_dev_acc = 0
    best_train_acc = -np.infty

    if args.cuda:
        model.cuda()

    if args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad([param for param in model.parameters() if param.requires_grad],
                                  lr=state['lr'], weight_decay=args.weight_decay)
        for group in optimizer.param_groups:
            for p in group['params']:
                optstate = optimizer.state[p]
                optstate['sum'] += args.Adagrad_init
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta([param for param in model.parameters() if param.requires_grad],
                                   lr=state['lr'])
    else:
        optimizer = optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=state['lr'])

    # load from checkpoint if provided
    if sys.argv[1]:
        checkpoint_dir = sys.argv[1]
        print('loading from checkpoint in {}'.format(checkpoint_dir))
        checkpoint = model_pipeline_pytorch.load_checkpoint(model, checkpoint=checkpoint_dir)
        if args.cuda:
            model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.state = defaultdict(dict, optimizer.state)
        state['lr'] = 0.0025
        print('resetting lr as {}'.format(state['lr']))
        model_pipeline_pytorch.change_learning_rate(optimizer,state['lr'])

    for epoch in range(args.epochs):
        dm.shuffle_train_data()

        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))
        if epoch % args.display_interval == 0:
            print('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = model_pipeline_pytorch.train(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            di=dm,
            args=args,
            loss_criterion=criterion,
        )
        dev_loss, dev_acc = model_pipeline_pytorch.test(
            model=model,
            epoch=epoch,
            di=dm,
            args=args,
            loss_criterion=criterion,
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

        if train_acc - best_train_acc < 1:
            state['lr'] *= args.learning_rate_decay
            model_pipeline_pytorch.change_learning_rate(optimizer, state['lr'])
            print('Epoch: [{} | {}] Update LR: {}'.format(
                epoch + 1, args.epochs, state['lr']))
        if train_acc > best_train_acc:
            best_train_acc = train_acc
