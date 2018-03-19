import sys
sys.path.append('./src')
sys.path.append('./src/models')
import torch.nn as nn
import numpy as np
import dataman.wrangle as wrangle
from dataman.squad_classif_data_manager import SquadDataManager
from models.seq2seq_model_pytorch import Seq2Seq
import models.model_pipeline_pytorch as model_pipeline_pytorch
import models.siamese_pytorch as siamese_pytorch
import models.squad_pytorch as squad_pytorch
from utils import dotdict
import torch
import torch.optim as optim
import sys
import logging

import models.load_embeddings as load_embeddings
import constants

logger = logging.getLogger(__name__)

NumEpochs = 10000
TrainEpochsPerTest = 500
nli_args = dotdict({
    'type': 'siamese',
    'encoder_type': 'rnn',
    'lr': 0.05,
    'learning_rate_decay': 0.99,
    'max_length': 50,
    'batch_size': 64,
    'batches_per_epoch': 3000,
    'test_batches_per_epoch': 500,
    'input_size': 300,
    'hidden_size': 2048,
    'n_layers': 1,
    'bidirectional': False,
    'embedding_size': 300,
    'fix_emb': True,
    'dp_ratio': 0.3,
    'd_out': 3,  # 3 classes
    'mlp_classif_hidden_size_list': [512, 512],
    'cuda': torch.cuda.is_available(),
})
nli_state = {k: v for k, v in nli_args.items()}

squad_args = dotdict({
    'type': 'siamese',
    'encoder_type': 'rnn',
    'lr': 0.05,
    'learning_rate_decay': 0.99,
    'max_length': 50,
    'batch_size': 128,
    'batches_per_epoch': 3000,
    'test_batches_per_epoch': 500,
    'input_size': 300,
    'hidden_size': 2048,
    'n_layers': 1,
    'bidirectional': False,
    'embedding_size': 300,
    'fix_emb': True,
    'dp_ratio': 0.3,
    'd_out': 2,  # 2 classes
    'mlp_classif_hidden_size_list': [512, 512],
    'cuda': torch.cuda.is_available(),
})
squad_state = {k: v for k, v in squad_args.items()}


if __name__ == "__main__":
    print(nli_args)

    nli_dm = wrangle.DataManager(nli_args)
    squad_dm = SquadDataManager(squad_args, vocab=nli_dm.vocab)
    nli_args.n_embed = nli_dm.vocab.n_words
    squad_dm.n_embed = squad_dm.vocab.n_words
    if nli_args.type == 'siamese':
        nli_model = siamese_pytorch.SiameseClassifier(config=nli_args)
        squad_model = squad_pytorch.SquadClassifier(
            config=squad_args,
            embed=nli_model.embed,
            encoder=nli_model.encoder,
        )
    else:
        raise Exception('model type not supported')

    nli_model.embed.weight.data = load_embeddings.load_embeddings(
        nli_dm.vocab, constants.EMBED_DATA_PATH, nli_args.embedding_size)

    best_nli_dev_acc = 0
    best_nli_train_acc = -np.infty

    nli_criterion = nn.NLLLoss()
    squad_criterion = nn.NLLLoss()

    for epoch in range(NumEpochs):
        nli_dm.shuffle_train_data()

        print('lr {}'.format(nli_state['lr']))
        nli_optimizer = optim.SGD(
            [param for param in nli_model.parameters()
             if param.requires_grad],
            lr=nli_state['lr'])
        squad_optimizer = optim.SGD(
            [param for param in squad_model.parameters()
             if param.requires_grad],
            lr=squad_state['lr'])
        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
            epoch + 1, NumEpochs, nli_state['lr']))

        if nli_args.cuda:
            nli_model.cuda()
            squad_model.cuda()

        _, squad_train_acc = model_pipeline_pytorch.train_squad(
            model=squad_model,
            optimizer=squad_optimizer,
            epoch=epoch,
            di=squad_dm,
            args=squad_args,
            loss_criterion=squad_criterion,
        )
        _, nli_train_acc = model_pipeline_pytorch.train(
            model=nli_model,
            optimizer=nli_optimizer,
            epoch=epoch,
            di=nli_dm,
            args=nli_args,
            loss_criterion=nli_criterion,
        )

        if epoch % TrainEpochsPerTest == 0 and epoch:
            _, nli_dev_acc = model_pipeline_pytorch.test(
                model=nli_model,
                epoch=epoch,
                di=nli_dm,
                args=nli_args,
                loss_criterion=nli_criterion,
            )
            _, squad_dev_acc = model_pipeline_pytorch.test_squad(
                model=squad_model,
                epoch=epoch,
                di=squad_dm,
                args=squad_args,
                loss_criterion=squad_criterion,
            )

            if nli_dev_acc > best_nli_dev_acc:
                print('New best model: {} vs {}'.format(
                    nli_dev_acc, best_nli_dev_acc))
                best_nli_dev_acc = nli_dev_acc
            print('Saving to checkpoint')
            model_pipeline_pytorch.save_checkpoint(
                nli_state={
                    'epoch': epoch + 1,
                    'nli_state_dict': nli_model.nli_state_dict(),
                    'acc': nli_dev_acc,
                    'best_acc': best_nli_dev_acc,
                    'optimizer': nli_optimizer.nli_state_dict()
                }, is_best=True)

        nli_state['lr'] *= nli_args.learning_rate_decay
        squad_state['lr'] *= squad_args.learning_rate_decay
