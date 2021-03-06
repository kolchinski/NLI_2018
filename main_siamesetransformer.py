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

logger = logging.getLogger(__name__)

args = dotdict({
    'encoder_type': 'transformer',
    'lr': 0.005,
    'learning_rate_decay': 0.8,
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
        criterion = nn.NLLLoss()
    else:
        model = Seq2SeqPytorch(args=args, vocab=dm.vocab)
        model.encoder.embedding.weight.data = load_embeddings.load_embeddings(
            dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)

    # number of parameters
    print("number of trainable parameters found {}".format(sum(
        param.nelement() for param in model.parameters()
        if param.requires_grad)))

    # load from checkpoint if provided
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
        print('loading from checkpoint in {}'.format(checkpoint_dir))
        model_pipeline_pytorch.load_checkpoint(model, checkpoint=checkpoint_dir)
        print('resetting lr as {}'.format(args.lr))


    best_dev_acc = 0
    best_train_acc = -np.infty

    for epoch in range(args.epochs):
        dm.shuffle_train_data()

        print('lr {}'.format(state['lr']))
        optimizer = optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=state['lr'])
        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, args.epochs, state['lr']))

        if args.cuda:
            model.cuda()
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
                },
                is_best=dev_acc > best_dev_acc,
            )
        if train_acc - best_train_acc < 3:
            state['lr'] *= args.learning_rate_decay
        if train_acc > best_train_acc:
            best_train_acc = train_acc
