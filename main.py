import sys
sys.path.append('./src')
import dataman.wrangle as wrangle
from models.seq2seq_model_pytorch import Seq2SeqPytorch
import models.model_pipeline_pytorch as model_pipeline_pytorch
from utils import dotdict
import torch
import torch.optim as optim
import sys
import logging

import models.load_embeddings as load_embeddings
import constants

logger = logging.getLogger(__name__)

args = dotdict({
    'lr': 0.01,
    'max_length': 5,
    'epochs': 5,
    'batch_size': 256,
    'batches_per_epoch': 1000,
    'test_batches_per_epoch': 10,
    'input_size': 300,
    'hidden_size': 128,
    'embedding_size': 300,
    'cuda': torch.cuda.is_available(),
})
state = {k: v for k, v in args.items()}


if __name__ == "__main__":
    print('use cuda: {}'.format(args.cuda))

    dm = wrangle.DataManager(args)
    model = Seq2SeqPytorch(args=args, vocab=dm.vocab)
    model.net.encoder.embedding.weight.data = load_embeddings.load_embeddings(
        dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)

    best_dev_acc = 0

    for epoch in range(args.epochs):
        optimizer = optim.Adam(
            [param for param in model.net.parameters() if param.requires_grad],
            lr=state['lr'])
        logger.info('\nEpoch: [{} | {}] LR: {}'.format(
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
            # TODO
