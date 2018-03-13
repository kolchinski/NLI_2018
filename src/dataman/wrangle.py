import constants
import json
import logging
import numpy as np
import sys
sys.path.append('../')
import torch
from torch.autograd import Variable
import models.vocab_pytorch as vocab_pytorch

logger = logging.getLogger(__name__)

NLILabelDict = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
}

class DataManager:

    def __init__(self, args):
        self.config = args
        self.max_len = self.config.max_length
        self.vocab = vocab_pytorch.Vocab()
        self.batch_size = args.batch_size
        print('loading train..')
        (
            self.train_sent1s_num, self.train_sent1s_len,
            self.train_sent2s_num, self.train_sent2s_len,
            self.train_sent1s_pos_embedinput, self.train_sent2s_pos_embedinput,
            self.train_ys, self.train_size
        ) = self.load_tok_data(constants.FULL_TRAIN_TOK_DATA_PATH, train=True)
        print('loading dev..')
        (
            self.dev_sent1s_num, self.dev_sent1s_len,
            self.dev_sent2s_num, self.dev_sent2s_len,
            self.dev_sent1s_pos_embedinput, self.dev_sent2s_pos_embedinput,
            self.dev_ys, self.dev_size
        ) = self.load_tok_data(constants.DEV_TOK_DATA_PATH)
        print('loading test..')
        (
            self.test_sent1s_num, self.test_sent1s_len,
            self.test_sent2s_num, self.test_sent2s_len,
            self.test_sent1s_pos_embedinput, self.test_sent2s_pos_embedinput,
            self.test_ys, self.test_size
        ) = self.load_tok_data(constants.TEST_TOK_DATA_PATH)
        self.curr_batch_train = 0
        self.curr_batch_dev = 0
        self.curr_batch_test = 0
        self.num_batch_train = self.train_size // self.batch_size
        self.num_batch_dev = self.dev_size // self.batch_size
        self.num_batch_test = self.test_size // self.batch_size

    def shuffle_train_data(self):
        permutation_np = np.random.permutation(len(self.train_ys))
        permutation = torch.LongTensor(permutation_np)
        self.train_sent1s_num = self.train_sent1s_num.index_select(0, permutation)
        self.train_sent1s_len = self.train_sent1s_len.index_select(0, permutation)
        self.train_sent2s_num = self.train_sent2s_num.index_select(0, permutation)
        self.train_sent2s_len = self.train_sent2s_len.index_select(0, permutation)
        self.train_ys = self.train_ys[permutation]

        self.curr_batch_train = 0

    def sample_train_batch(
        self,
        use_cuda,
        encoder_embed=None,
        decoder_embed=None,
    ):
        sample_idx = self.curr_batch_train * self.batch_size
        train_sent1s_num = self.train_sent1s_num[sample_idx: sample_idx + self.batch_size]
        train_sent1s_len = self.train_sent1s_len[sample_idx: sample_idx + self.batch_size]
        train_sent1s_pos_embedinput = self.train_sent1s_pos_embedinput[
            sample_idx: sample_idx + self.batch_size]
        train_sent2s_num = self.train_sent2s_num[sample_idx: sample_idx + self.batch_size]
        train_sent2s_len = self.train_sent2s_len[sample_idx: sample_idx + self.batch_size]
        train_sent2s_pos_embedinput = self.train_sent2s_pos_embedinput[
            sample_idx: sample_idx + self.batch_size]
        targets_tensor = self.train_ys[sample_idx: sample_idx + self.batch_size]

        if self.config.encoder_type == 'transformer':
            return (
                train_sent1s_num,
                train_sent1s_pos_embedinput,
                train_sent2s_num,
                train_sent2s_pos_embedinput,
                targets_tensor,
            )

        seq1_packed_tensor, seq1_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=train_sent1s_num,
            seq_lengths=train_sent1s_len,
            embed=encoder_embed,
            use_cuda=use_cuda,
        )
        seq2_packed_tensor, seq2_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=train_sent2s_num,
            seq_lengths=train_sent2s_len,
            embed=decoder_embed,
            use_cuda=use_cuda,
        )
        self.curr_batch_train = (self.curr_batch_train + 1) % self.num_batch_train

        return (
            seq1_packed_tensor,  # [batch_size, seq_len]
            seq2_packed_tensor,  # [batch_size, seq_len]
            seq1_idx_unsort,
            seq2_idx_unsort,
            Variable(targets_tensor),  # [batch_size,]
        )

    def sample_dev_batch(
        self,
        encoder_embed,
        decoder_embed,
        use_cuda,
    ):
        sample_idx = self.curr_batch_dev * self.batch_size
        dev_sent1s_num = self.dev_sent1s_num[sample_idx: sample_idx + self.batch_size]
        dev_sent1s_len = self.dev_sent1s_len[sample_idx: sample_idx + self.batch_size]
        dev_sent1s_pos_embedinput = self.dev_sent1s_pos_embedinput[
            sample_idx: sample_idx + self.batch_size]
        dev_sent2s_num = self.dev_sent2s_num[sample_idx: sample_idx + self.batch_size]
        dev_sent2s_len = self.dev_sent2s_len[sample_idx: sample_idx + self.batch_size]
        dev_sent2s_pos_embedinput = self.dev_sent1s_pos_embedinput[
            sample_idx: sample_idx + self.batch_size]
        targets_tensor = self.dev_ys[sample_idx: sample_idx + self.batch_size]

        if self.config.encoder_type == 'transformer':
            return (
                dev_sent1s_num,
                dev_sent1s_pos_embedinput,
                dev_sent2s_num,
                dev_sent2s_pos_embedinput,
                targets_tensor,
            )

        seq1_packed_tensor, seq1_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=dev_sent1s_num,
            seq_lengths=dev_sent1s_len,
            embed=encoder_embed,
            use_cuda=use_cuda,
        )
        seq2_packed_tensor, seq2_idx_unsort = self.vocab.get_packedseq_from_sent_batch(
            seq_tensor=dev_sent2s_num,
            seq_lengths=dev_sent2s_len,
            embed=decoder_embed,
            use_cuda=use_cuda,
        )
        self.curr_batch_dev = (self.curr_batch_dev + 1) % self.num_batch_dev

        return (
            seq1_packed_tensor,  # [batch_size, seq_len]
            seq2_packed_tensor,  # [batch_size, seq_len]
            seq1_idx_unsort,
            seq2_idx_unsort,
            Variable(targets_tensor),  # [batch_size,]
        )


    def load_tok_data(self, path, train=False):
        sent1s, sent2s, targets = [], [], []
        path_label = path+'labels'
        path_s1 = path+'s1'
        path_s2 = path+'s2'
        with open(path_label) as f:
            lines = f.readlines()
            n_rows = len(lines)
            for l in lines:
                lab = l.rstrip()
                targets.append(NLILabelDict[lab])
        with open(path_s1) as f:
            lines = f.readlines()
            if len(lines) != n_rows:
                raise RuntimeError(
                    "Sent1 tokens has {} lines, but labels have {} lines. They must have "
                    "the same number of lines.".format(len(lines), n_rows))
            for l in lines:
                sent1 = l.rstrip().lower()
                sent1s.append(sent1)
                if True:  # if train:
                    # this is slightly cheating (adding dev vocab)
                    # but convenient so we dont load the entire GloVe vocab
                    self.vocab.addSentence(sent1)
        with open(path_s2) as f:
            lines = f.readlines()
            if len(lines) != n_rows:
                raise RuntimeError(
                    "Sent2 tokens has {} lines, but labels have {} lines. They must have "
                    "the same number of lines.".format(len(lines), n_rows))
            for l in lines:
                sent2 = l.rstrip().lower()
                sent2s.append(sent2)
                if True:  # if train:
                    # this is slightly cheating (adding dev vocab)
                    # but convenient so we dont load the entire GloVe vocab
                    self.vocab.addSentence(sent2)

        print('read {} pairs'.format(n_rows))
        if True:
            print('vocab size: {}'.format(self.vocab.n_words))

        # positional embeddings
        def get_pos_embedinputinput(sents):
            pos_embedinput_arr = np.array([
                [pos + 1 if w != vocab_pytorch.PAD_token else 0
                 for pos, w in enumerate(sent)]
                for sent in sents
            ])
            pos_embedinput_tensor = Variable(
                torch.LongTensor(pos_embedinput_arr), volatile=train)
            return pos_embedinput_tensor

        sent1_pos_embedinput_tensor = get_pos_embedinputinput(sent1s)  # [batch_size, max_len]
        sent2_pos_embedinput_tensor = get_pos_embedinputinput(sent2s)

        targets = torch.from_numpy(
            np.array(targets, dtype=np.int64)  # expect LongTensor
        )

        print('numberizing')
        sent1s_num = [self.vocab.numberize_sentence(s) for s in sent1s]
        sent2s_num = [self.vocab.numberize_sentence(s) for s in sent2s]
        print('done.')

        # load numberized into tensors
        sent1_bin_tensor, sent1_len_tensor = self.get_numberized_tensor(
            sent1s_num)
        sent2_bin_tensor, sent2_len_tensor = self.get_numberized_tensor(
            sent2s_num)

        return (
            sent1_bin_tensor, sent1_len_tensor,
            sent2_bin_tensor, sent2_len_tensor,
            sent1_pos_embedinput_tensor, sent2_pos_embedinput_tensor,
            targets, n_rows
        )

    def get_numberized_tensor(self, sent_num):
        """ sent_num: list of lists of word ids """
        dat_size = len(sent_num)
        bin_tensor = torch.LongTensor(dat_size, self.max_len)
        bin_tensor.fill_(vocab_pytorch.PAD_token)
        # slen_max = htable_params.max_len
        slen_tensor = torch.IntTensor(dat_size,)

        b = 0
        for sent_wordids in sent_num:
            slen = min(len(sent_wordids), self.max_len)
            slen_tensor[b] = slen
            for w in range(slen):
                wordid = sent_wordids[w]
                bin_tensor[b, slen - 1 - w] = wordid  # fill in reverse
            b += 1

        return bin_tensor, slen_tensor
