import constants
import json
import logging
import sys
sys.path.append('../models')
import vocab_pytorch

logger = logging.getLogger(__name__)

NLILabelDict = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
}

class DataManager:
    def __init__(self):
        self.vocab = vocab_pytorch.Vocab()
        print('loading train..')
        self.train_sent1s, self.train_sent2s, self.train_ys = \
            self.load_data(constants.SMALL_TRAIN_DATA_PATH, train=True)
        print('loading dev..')
        self.dev_sent1s, self.dev_sent2s, self.dev_ys = \
            self.load_data(constants.DEV_DATA_PATH)
        print('loading test..')
        self.test_sent1s, self.test_sent2s, self.test_ys = \
            self.load_data(constants.TEST_DATA_PATH)

    def sample_train_batch(self, batch_size):
        # TODO
        return (
            self.train_sent1s[:batch_size],
            self.train_sent2s[:batch_size],
            self.train_ys[:batch_size],
        )

    def load_data(self, path, train=False):
        sent1s, sent2s, ys = [], [], []
        with open(path, 'r') as f:
            for l in f.readlines():
                dat = json.loads(l)
                lab = dat['gold_label']
                if lab in NLILabelDict:
                    sent1s.append(dat['sentence1'])
                    sent2s.append(dat['sentence2'])
                    ys.append(NLILabelDict[lab])
                    if train:
                        self.vocab.addSentence(dat['sentence1'])
                        self.vocab.addSentence(dat['sentence2'])
        print('read {} pairs'.format(len(sent1s)))
        if train:
            print('vocab size: {}'.format(self.vocab.n_words))
        return sent1s, sent2s, ys
