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
import time
from pytorch_classification.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


import models.load_embeddings as load_embeddings
import constants

args = dotdict({
    'encoder_type': 'transformer',
    'lr': 0.01,
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

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=args.test_batches_per_epoch)
    batch_idx = 0

    for _ in range(dm.num_batch_test):
        # sample batch
        if args.encoder_type == 'transformer':
            sent1, sent1_posembinput, sent2, sent2_posembinput, targets = \
                dm.get_next_test_batch(use_cuda=args.cuda)
            unsort1, unsort2 = None, None
            encoder_init_hidden = None
        elif args.encoder_type == 'rnn':
            sent1, sent2, unsort1, unsort2, targets = dm.get_next_test_batch(
                encoder_embed=model.embed,
                decoder_embed=model.embed,
                use_cuda=args.cuda,
            )
            sent1_posembinput, sent2_posembinput = None, None
            encoder_init_hidden = model.encoder.initHidden(
                batch_size=args.batch_size)
        else:
            raise Exception('encoder_type not supported {}'.format(
                args.encoder_type))
        if args.cuda:
            model = model.cuda()
            targets = targets.cuda(async=True)
            if args.encoder_type == 'transformer':
                sent1 = sent1.cuda()
                sent2 = sent2.cuda()
                sent1_posembinput = sent1_posembinput.cuda()
                sent2_posembinput = sent2_posembinput.cuda()
            if args.encoder_type == 'rnn':
                if len(encoder_init_hidden):
                    encoder_init_hidden = [x.cuda() for x in encoder_init_hidden]
                else:
                    encoder_init_hidden = encoder_init_hidden.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        softmax_outputs = model(
            encoder_init_hidden=encoder_init_hidden,
            encoder_input=sent1,
            encoder_pos_emb_input=sent1_posembinput,
            encoder_unsort=unsort1,
            decoder_input=sent2,
            decoder_pos_emb_input=sent2_posembinput,
            decoder_unsort=unsort2,
            batch_size=args.batch_size,
        )

        # measure accuracy and record loss
        acc_batch = model_pipeline_pytorch.compute_accuracy(
            outputs=softmax_outputs.data,
            targets=targets.data,
        )
        acc.update(acc_batch, args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s'\
        '| Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc:.3f}'\
            .format(
                batch=batch_idx,
                size=args.test_batches_per_epoch,
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                acc=acc.avg,
            )
        bar.next()
    bar.finish()
