# python 3
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

PAD_token = 0
EOS_token = 1
UNK_token = 2


class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: '<pad>', 1: "EOS", 2: "UNK"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def numberize_sentence(self, sentence):
        numberized = [
            self.word2index[word]
            if word in self.word2index else UNK_token
            for word in sentence.split(' ')
        ]
        numberized.append(EOS_token)
        return numberized

    def get_packedseq_from_sent_batch(
        self,
        seq_tensor,
        seq_lengths,
        embed,
        use_cuda,
    ):
        if use_cuda:
            seq_lengths = seq_lengths.cuda()
            seq_tensor = seq_tensor.cuda()

        # sort by length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        seq_tensor = seq_tensor.transpose(0, 1)  # [seq_len, batch_size]
        idx_unsort = Variable(torch.from_numpy(
            np.argsort(perm_idx.cpu().numpy())
        ))
        if use_cuda:
            idx_unsort = idx_unsort.cuda()

        # embed
        seq_tensor = embed(seq_tensor)

        # pack
        seq_pack_tensor = nn.utils.rnn.pack_padded_sequence(
            seq_tensor.cuda() if use_cuda else seq_tensor,
            seq_lengths.cpu().numpy(),
        )

        return seq_pack_tensor, idx_unsort
