# python 3
import torch
from torch.autograd import Variable

SOS_token = 0
EOS_token = 1


class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

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


def indexesFromSentence(word2index, sentence):
    return [word2index[word] for word in sentence.split(' ')]


def variableFromSentence(word2index, sentence, use_cuda):
    indexes = indexesFromSentence(word2index, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(word2index, pair):
    input_variable = variableFromSentence(word2index, pair[0])
    target_variable = variableFromSentence(word2index, pair[1])
    return (input_variable, target_variable)
