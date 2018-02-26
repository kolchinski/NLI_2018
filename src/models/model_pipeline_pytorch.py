import sys
sys.path.append('../../../')

import os
import shutil
import time
from pytorch_classification.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
#from pytorch_classification.utils.progress.progress.bar import Bar
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sklearn.metrics


def compute_accuracy(outputs, targets):
    acc = accuracy(outputs, targets, topk=(1,))
    return acc[0][0]


def train(model, optimizer, epoch, di, args, loss_criterion):
    model.train()  # switch to train mode

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=args.batches_per_epoch)
    batch_idx = 0

    while batch_idx < args.batches_per_epoch:
        # sample batch
        if False:
            sent1, sent2, unsort1, unsort2, targets = di.sample_train_batch(
                encoder_embed=model.encoder.embedding,
                decoder_embed=model.decoder.embedding,
                use_cuda=args.cuda,
            )
        else:
            sent1, sent2, unsort1, unsort2, targets = di.sample_train_batch(
                encoder_embed=model.embed,
                decoder_embed=model.embed,
                use_cuda=args.cuda,
            )
        encoder_init_hidden = model.encoder.initHidden(
            batch_size=args.batch_size)

        if args.cuda:
            model = model.cuda()
            targets = targets.cuda(async=True)
            if len(encoder_init_hidden):
                encoder_init_hidden = [x.cuda() for x in encoder_init_hidden]
            else:
                encoder_init_hidden = encoder_init_hidden.cuda()
            loss_criterion = loss_criterion.cuda()

        # measure data loading timeult
        data_time.update(time.time() - end)

        # compute output
        softmax_outputs = model(
            encoder_init_hidden=encoder_init_hidden,
            encoder_input=sent1,
            encoder_unsort=unsort1,
            decoder_input=sent2,
            decoder_unsort=unsort2,
            batch_size=args.batch_size,
        )
        loss = loss_criterion(softmax_outputs, targets)

        # measure accuracy and record loss
        acc_batch = compute_accuracy(
            outputs=softmax_outputs.data,
            targets=targets.data,
        )
        acc.update(acc_batch, args.batch_size)
        losses.update(loss.data[0], len(sent1))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s'\
        '| Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc:.3f}'\
            .format(
                batch=batch_idx,
                size=args.batches_per_epoch,
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                acc=acc.avg,
            )
        bar.next()
    bar.finish()

    return losses.avg, acc.avg

def test(model, epoch, di, args, loss_criterion):
    global best_acc

    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=args.test_batches_per_epoch)
    batch_idx = 0

    while batch_idx < args.test_batches_per_epoch:
        # sample batch
        if False:
            sent1, sent2, unsort1, unsort2, targets = di.sample_train_batch(
                encoder_embed=model.encoder.embedding,
                decoder_embed=model.decoder.embedding,
                use_cuda=args.cuda,
            )
        else:
            sent1, sent2, unsort1, unsort2, targets = di.sample_train_batch(
                encoder_embed=model.embed,
                decoder_embed=model.embed,
                use_cuda=args.cuda,
            )
        encoder_init_hidden = model.encoder.initHidden(
            batch_size=args.batch_size)
        if args.cuda:
            model = model.cuda()
            targets = targets.cuda(async=True)
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
            encoder_unsort=unsort1,
            decoder_input=sent2,
            decoder_unsort=unsort2,
            batch_size=args.batch_size,
        )
        loss = loss_criterion(softmax_outputs, targets)

        # measure accuracy and record loss
        acc_batch = compute_accuracy(
            outputs=softmax_outputs.data,
            targets=targets.data,
        )
        acc.update(acc_batch, args.batch_size)
        losses.update(loss.data[0], len(sent1))

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
    return losses.avg, acc.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}"
              .format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(
            filepath,
            os.path.join(checkpoint, 'model_best.pth.tar'),
        )


def load_checkpoint(model, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
    filepath = os.path.join(checkpoint, 'model_best.pth.tar')
    if not os.path.exists(filepath):
        raise("No best model in path {}".format(checkpoint))
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def adjust_learning_rate(optimizer, epoch, args, state):
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
