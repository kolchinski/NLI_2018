import sys
sys.path.append('../../../')

import os
import shutil
import time
from pytorch_classification.utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from pytorch_classification.utils.progress.progress.bar import Bar
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sklearn.metrics


def eval(outputs, targets, args, thres=0.5, eps=1e-9):
    acc = 0
    return acc


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
        sent1, sent2, lab = di.sample_train_batch(args.batch_size)
        input1 = di.vocab.get_packedseq_from_sent_batch(
            sent1, embed=model.encoder.embedding, use_cuda=args.cuda)
        input2 = di.vocab.get_packedseq_from_sent_batch(
            sent2, embed=model.decoder.embedding, use_cuda=args.cuda)
        targets = Variable(torch.from_numpy(np.array(lab)))
        if args.cuda:
            targets = targets.cuda(async=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        softmax_outputs = model(input1, input2)
        loss = loss_criterion(softmax_outputs, targets)

        # measure accuracy and record loss
        acc_batch = eval(softmax_outputs.data, targets.data, args)
        acc.update(acc_batch)
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
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s'
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
        next(bar)
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
        shutil.copyfile(filepath,
                        os.path.join(checkpoint, 'model_best.pth.tar'))


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
