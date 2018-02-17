import sys
sys.path.append('../../../')

import os
import shutil
import time
from pytorch_classification.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sklearn.metrics


def eval(outputs, targets, args, thres=0.5, eps=1e-9):
    index = torch.LongTensor([1])
    if args.cuda:
        index = index.cuda()
    outputs_class = torch.index_select(outputs, 1, index=index).view(-1)
    outputs_class = torch.exp(outputs_class)
    preds = torch.ge(outputs_class.float(), thres).float()
    targets = targets.float()
    true_positive = (preds * targets).sum()
    precis = true_positive / (preds.sum() + eps)
    recall = true_positive / (targets.sum() + eps)
    f1 = 2 * precis * recall / (precis + recall + eps)
    return (precis, recall, f1)


def train(model, optimizer, epoch, di, args, loss_criterion):
    model.train()  # switch to train mode

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mask_losses = AverageMeter()
    precis = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=args.batches_per_epoch)
    batch_idx = 0

    all_preds = np.array([])
    all_targets = np.array([])

    while batch_idx < args.batches_per_epoch:
        sent1, sent2, lab = di.sample_train_batch(args.batch_size)
        seq_batch = torch.from_numpy(seq_batch)
        gene_batch = torch.FloatTensor(gene_batch)
        targets = torch.from_numpy(target_batch)
        locus_mean_batch = torch.FloatTensor(locus_mean_batch)

        # measure data loading time
        data_time.update(time.time() - end)

        # predict
        if args.cuda:
            seq_batch, gene_batch, targets, locus_mean_batch = seq_batch.contiguous().cuda(), gene_batch.contiguous().cuda(), targets.cuda(async=True), locus_mean_batch.contiguous().cuda()
        seq_batch, gene_batch, targets, locus_mean_batch = Variable(seq_batch), Variable(gene_batch), Variable(targets), Variable(locus_mean_batch)

        # compute output
        outputs, masks = model(seq_batch, gene_batch, locus_mean_batch)
        loss = loss_criterion(outputs, targets)

        # concat to all_preds, all_targets
        index = Variable(torch.LongTensor([1]))
        if args.cuda:
            index = index.cuda()
        all_preds = np.concatenate((all_preds, torch.index_select(outputs, 1, index=index).view(-1).cpu().data.numpy()))
        all_targets = np.concatenate((all_targets, targets.cpu().data.numpy()))

        # measure accuracy and record loss
        p, r, f = eval(outputs.data, targets.data, args)
        precis.update(p, seq_batch.size(0))
        recall.update(r, seq_batch.size(0))
        f1.update(f, seq_batch.size(0))
        losses.update(loss.data[0], seq_batch.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | MaskL: {mloss:.4f} | prec: {precis:.3f} | rec: {recall:.3f}'.format(
                    batch=batch_idx,
                    size=args.batches_per_epoch,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    mloss=mask_losses.avg,
                    precis=precis.avg,
                    recall=recall.avg,
                    )
        bar.next()
    bar.finish()

    # compute train auprc/auc for direct comparison to test
    train_auprc = sklearn.metrics.average_precision_score(all_targets, all_preds)
    train_auc = sklearn.metrics.roc_auc_score(all_targets, all_preds)
    print('train auprc: {auprc: .3f} | train auc: {auc: .3f}'.format(
        auprc=train_auprc,
        auc=train_auc,
    ))

    return (losses.avg, f1.avg)


def test(model, optimizer, epoch, di, args, loss_criterion, mask_criterion):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mask_losses = AverageMeter()
    precis = AverageMeter()
    recall = AverageMeter()
    f1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=args.batches_per_test_epoch)
    batch_idx = 0
    all_preds = np.array([])
    all_targets = np.array([])

    while batch_idx < args.batches_per_test_epoch:
        # measure data loading time
        data_time.update(time.time() - end)
        seq_batch, gene_batch, target_batch, locus_mean_batch= di.sample_validation_batch(args.batch_size)
        seq_batch = torch.from_numpy(seq_batch)
        gene_batch = torch.FloatTensor(gene_batch)
        targets = torch.from_numpy(target_batch)
        locus_mean_batch = torch.FloatTensor(locus_mean_batch)
        if args.cuda:
            seq_batch, gene_batch, targets, locus_mean_batch = seq_batch.contiguous().cuda(), gene_batch.contiguous().cuda(), targets.cuda(), locus_mean_batch.contiguous().cuda()
        seq_batch, gene_batch, targets, locus_mean_batch = Variable(seq_batch, volatile=True), Variable(gene_batch, volatile=True), Variable(targets), Variable(locus_mean_batch, volatile=True)

        # compute output
        outputs, masks = model(seq_batch, gene_batch, locus_mean_batch)
        loss = loss_criterion(outputs, targets)
        mask_loss = mask_criterion(masks)

        # concat to all_preds, all_targets
        index = Variable(torch.LongTensor([1]))
        if args.cuda:
            index = index.cuda()
        all_preds = np.concatenate((all_preds, torch.index_select(outputs, 1, index=index).view(-1).cpu().data.numpy()))
        all_targets = np.concatenate((all_targets, targets.cpu().data.numpy()))

        # measure accuracy and record loss
        p, r, f = eval(outputs.data, targets.data, args)
        auprc = sklearn.metrics.average_precision_score(all_targets, all_preds)
        auc = sklearn.metrics.roc_auc_score(all_targets, all_preds)
        precis.update(p, seq_batch.size(0))
        recall.update(r, seq_batch.size(0))
        f1.update(f, seq_batch.size(0))
        losses.update(loss.data[0], seq_batch.size(0))
        mask_losses.update(mask_loss.data[0], seq_batch.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_idx += 1
        # plot progress
        bar.suffix  = '({batch}/{size}) | Loss: {loss:.4f} | Mask Loss: {mloss:.4f} | auprc: {auprc:.3f} | auc: {auc:.3f}'.format(
                    batch=batch_idx,
                    size=args.batches_per_test_epoch,
                    loss=losses.avg,
                    mloss=mask_losses.avg,
                    auprc=auprc,
                    auc=auc,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, auprc)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def load_checkpoint(model, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
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
