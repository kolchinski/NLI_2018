import numpy as np
import copy
from senteval import utils

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


'''
A = softmax(W2 tanh(W1 H^T))
'''


class SelfAttentionModel(nn.Module):

    def __init__(self, config, embed, encoder):
        super(SelfAttentionModel, self).__init__()
        self.config = config
        self.embed = embed
        self.encoder = encoder

        self.num_units = self.config.hidden_size
        self.num_out_units = self.num_units * config.self_attn_outer_size
        if self.config.bidirectional:
            self.num_units *= 2
        self.w1_selfattn = nn.Parameter(
            torch.FloatTensor(
                config.self_attn_inner_size, self.num_units
            ).normal_(0, 0.1),
            requires_grad=True,
        )
        self.w2_selfattn = nn.Parameter(
            torch.FloatTensor(
                config.self_attn_outer_size, config.self_attn_inner_size
            ).normal_(0, 0.1),
            requires_grad=True,
        )

    def forward(self, encoder_outputs, batch_len):
        encoder_out_batchfirst = encoder_outputs.permute(1, 0, 2)
        encoder_out_tr = encoder_outputs.permute(2, 0, 1)  # [bs, nunits, slen]

        pre_softmax = torch.matmul(
            self.s2w2_selfattn,
            torch.nn.Tanh(torch.matmul(self.w1_selfattn, encoder_out_tr)),
        )
        A_selfattn = torch.Softmax(pre_softmax, dim=1)

        # [bs, attn_outer_size, num_units]
        encoder_outputs_attn = torch.matmul(A_selfattn, encoder_out_batchfirst)

        # [bs, attn_outer_size * num_units]
        return encoder_outputs_attn.view(-1, self.num_out_units)


class SelfAttentionClassifier(object):
    def __init__(self, params, inputdim, nclasses, embed, encoder,
                 l2reg=0., batch_size=64, seed=1111, cudaEfficient=False):
        super(self.__class__, self).__init__(inputdim, nclasses, l2reg,
                                             batch_size, seed, cudaEfficient)
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 4 if "epoch_size" not in params else \
            params["epoch_size"]
        self.max_epoch = 200 if "max_epoch" not in params else \
            params["max_epoch"]
        self.dropout = 0. if "dropout" not in params else \
            params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else \
            params["batch_size"]

        self.self_attn = SelfAttentionModel(
            config=params.config,
            embed=embed,
            encoder=encoder,
        )

        self.model = nn.Sequential(
            self.self_attn,
            nn.Linear(self.inputdim, params["nhid"]),
            nn.Dropout(p=self.dropout),
            nn.Sigmoid(),
            nn.Linear(params["nhid"], self.nclasses),
            ).cuda()

        self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]['weight_decay'] = self.l2reg

    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split*len(X)):]
            devidx = permutation[0:int(validation_split*len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        if not self.cudaEfficient:
            trainX = torch.FloatTensor(trainX).cuda()
            trainy = torch.LongTensor(trainy).cuda()
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        else:
            trainX = torch.FloatTensor(trainX)
            trainy = torch.LongTensor(trainy)
            devX = torch.FloatTensor(devX)
            devy = torch.LongTensor(devy)

        return trainX, trainy, devX, devy

    def fit(self, X, y, validation_data=None, validation_split=None,
            early_stop=True):
        self.nepoch = 0
        bestaccuracy = -1
        stop_train = False
        early_stop_count = 0

        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(X, y, validation_data,
                                                        validation_split)

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
            accuracy = self.score(devX, devy)
            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        return bestaccuracy

    def trainepoch(self, X, y, epoch_size=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = torch.LongTensor(permutation[i:i + self.batch_size])
                if isinstance(X, torch.cuda.FloatTensor):
                    idx = idx.cuda()
                Xbatch = Variable(X.index_select(0, idx))
                ybatch = Variable(y.index_select(0, idx))
                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data[0])
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy):
        self.model.eval()
        correct = 0
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX).cuda()
            devy = torch.LongTensor(devy).cuda()
        for i in range(0, len(devX), self.batch_size):
            Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
            ybatch = Variable(devy[i:i + self.batch_size], volatile=True)
            if self.cudaEfficient:
                Xbatch = Xbatch.cuda()
                ybatch = ybatch.cuda()
            output = self.model(Xbatch)
            pred = output.data.max(1)[1]
            correct += pred.long().eq(ybatch.data.long()).sum()
        accuracy = 1.0*correct / len(devX)
        return accuracy

    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX).cuda()
        yhat = np.array([])
        for i in range(0, len(devX), self.batch_size):
            Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
            output = self.model(Xbatch)
            yhat = np.append(yhat,
                             output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        for i in range(0, len(devX), self.batch_size):
            Xbatch = Variable(devX[i:i + self.batch_size], volatile=True)
            vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
            if not probas:
                probas = vals
            else:
                probas = np.concatenate(probas, vals, axis=0)
        return probas
