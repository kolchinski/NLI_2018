# python 3
import torch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

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

    def get_packedseq_from_sent_batch(self, sent_batch, embed, use_cuda):
        numberized_batch = []
        for sent in sent_batch:
            numberized = self.numberize_sentence(sent)
            numberized_batch.append(numberized)
        print('numberized example: {} for sent: \"{}\"'.format(
            numberized, sent_batch[-1]))
        seq_lengths = torch.LongTensor(map(len, numberized_batch))
        seq_tensor = Variable(torch.zeros(
            (len(numberized_batch), seq_lengths.max()))).long()
        if use_cuda:
            seq_lengths.cuda()
            seq_tensor.cuda()

        # sort by length
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        seq_tensor = seq_tensor.transpose(0, 1)  # [seq_len, batch_size]

        # embed
        seq_tensor = embed(seq_tensor)

        # pack
        seq_pack_tensor = nn.utils.rnn.pack_padded_sequence(
            seq_tensor.cuda() if use_cuda else seq_tensor,
            seq_lengths.cpu().numpy(),
        )

        return seq_pack_tensor
