import sys
sys.path.append('../')
import src.constants as constants
import src.models.vocab_pytorch as vocab_pytorch

import logging
import numpy as np

import torch
from torch.autograd import Variable
from wrangle import DataManager

logger = logging.getLogger(__name__)


class SquadDataManager(DataManager):

    def __init__(self, args, vocab):
        self.config = args
        self.max_len = self.config.max_length
        self.vocab = vocab
        self.batch_size = args.batch_size
        print('loading train..')
        (
            self.train_a1s, self.train_a1s_len,
            self.train_a2s, self.train_a2s_len,
            self.train_qs, self.train_q2s_len,
            self.train_ys, self.train_size
        ) = self.load_tok_data(
            constants.SQUAD_FULL_TRAIN_DATA_PATH, train=True)
        print('loading dev..')
        (
            self.dev_a1s, self.dev_a1s_len,
            self.dev_a2s, self.dev_a2s_len,
            self.dev_qs, self.dev_q2s_len,
            self.dev_ys, self.dev_size
        ) = self.load_tok_data(constants.SQUAD_DEV_DATA_PATH)
        self.curr_batch_train = 0
        self.curr_batch_dev = 0
        self.num_batch_train = self.train_size // self.batch_size
        self.num_batch_dev = self.dev_size // self.batch_size
        self.num_batch_test = self.test_size // self.batch_size

    def shuffle_train_data(self):
        permutation_np = np.random.permutation(len(self.train_ys))
        permutation = torch.LongTensor(permutation_np)
        self.train_a1s = self.train_a1s.index_select(0, permutation)
        self.train_a1s_len = self.train_a1s_len.index_select(0, permutation)
        self.train_a2s = self.train_a2s.index_select(0, permutation)
        self.train_a2s_len = self.train_a2s_len.index_select(0, permutation)
        self.train_qs = self.train_qs.index_select(0, permutation)
        self.train_q2s_len = self.train_q2s_len.index_select(0, permutation)
        self.train_ys = self.train_ys[permutation]

        self.curr_batch_train = 0

    def sample_train_batch(
        self,
        use_cuda,
        encoder_embed=None,
        decoder_embed=None,
    ):
        sample_idx = self.curr_batch_train * self.batch_size
        sample_idx = range(sample_idx, sample_idx + self.batch_size)
        train_a1s = self.train_a1s[sample_idx]
        train_a1s_len = self.train_a1s_len[sample_idx]
        train_a2s = self.train_a2s[sample_idx]
        train_a2s_len = self.train_a2s_len[sample_idx]
        train_qs = self.train_qs[sample_idx]
        train_qs_len = self.train_qs_len[sample_idx]

        targets_tensor = self.train_ys[sample_idx]

        self.curr_batch_train = \
            (self.curr_batch_train + 1) % self.num_batch_train

        a1_packed_tensor, a1_idx_unsort = self.\
            vocab.get_packedseq_from_sent_batch(
                seq_tensor=train_a1s,
                seq_lengths=train_a1s_len,
                embed=encoder_embed,
                use_cuda=use_cuda,
            )
        a2_packed_tensor, a2_idx_unsort = self.\
            vocab.get_packedseq_from_sent_batch(
                seq_tensor=train_a2s,
                seq_lengths=train_a2s_len,
                embed=encoder_embed,
                use_cuda=use_cuda,
            )
        q_packed_tensor, q_idx_unsort = self.\
            vocab.get_packedseq_from_sent_batch(
                seq_tensor=train_qs,
                seq_lengths=train_qs_len,
                embed=encoder_embed,
                use_cuda=use_cuda,
            )

        return (
            a1_packed_tensor,  # [batch_size, seq_len]
            a1_idx_unsort,  # [batch_size,]
            a2_packed_tensor,
            a2_idx_unsort,
            q_packed_tensor,
            q_idx_unsort,
            Variable(targets_tensor),
        )

    def sample_dev_batch(
        self,
        use_cuda,
        encoder_embed=None,
        decoder_embed=None,
    ):
        sample_idx = self.curr_batch_dev * self.batch_size
        sample_idx = range(sample_idx, sample_idx + self.batch_size)
        dev_a1s = self.dev_a1s[sample_idx]
        dev_a1s_len = self.dev_a1s_len[sample_idx]
        dev_a2s = self.dev_a2s[sample_idx]
        dev_a2s_len = self.dev_a2s_len[sample_idx]
        dev_qs = self.dev_qs[sample_idx]
        dev_qs_len = self.dev_qs_len[sample_idx]
        targets_tensor = self.dev_ys[sample_idx]

        self.curr_batch_dev = \
            (self.curr_batch_dev + 1) % self.num_batch_dev

        a1_packed_tensor, a1_idx_unsort = self.\
            vocab.get_packedseq_from_sent_batch(
                seq_tensor=dev_a1s,
                seq_lengths=dev_a1s_len,
                embed=encoder_embed,
                use_cuda=use_cuda,
            )
        a2_packed_tensor, a2_idx_unsort = self.\
            vocab.get_packedseq_from_sent_batch(
                seq_tensor=dev_a2s,
                seq_lengths=dev_a2s_len,
                embed=encoder_embed,
                use_cuda=use_cuda,
            )
        q_packed_tensor, q_idx_unsort = self.\
            vocab.get_packedseq_from_sent_batch(
                seq_tensor=dev_qs,
                seq_lengths=dev_qs_len,
                embed=encoder_embed,
                use_cuda=use_cuda,
            )

        return (
            a1_packed_tensor,  # [batch_size, seq_len]
            a1_idx_unsort,  # [batch_size,]
            a2_packed_tensor,
            a2_idx_unsort,
            q_packed_tensor,
            q_idx_unsort,
            Variable(targets_tensor),
        )

    def load_tok_data(self, path, train=False):
        a1s, a2s, qs, targets = [], [], []
        with open(path) as f:
            n_rows = 0
            for line in f.readlines():
                yi, a1, a2, qi = line.split('\t')
                a1s.append(a1)
                a2s.append(a2)
                qs.append(qi)
                targets.append(yi)

                # add to vocab
                self.vocab.addSentence(a1)
                self.vocab.addSentence(a2)
                self.vocab.addSentence(qi)

                n_rows += 1
            print('read {} rows'.format(n_rows))
        targets = torch.from_numpy(
            np.array(targets, dtype=np.int64)  # expect LongTensor
        )

        print('numberizing')
        a1s_num, a1_bin_tensor, a1_len_tensor = self.\
            numberize_sents_to_tensor(a1s)
        a2s_num, a2_bin_tensor, a2_len_tensor = self.\
            numberize_sents_to_tensor(a2s)
        qs_num, q_bin_tensor, q_len_tensor = self.\
            numberize_sents_to_tensor(qs)
        print('done.')

        return (
            a1_bin_tensor, a1_len_tensor,
            a2_bin_tensor, a2_len_tensor,
            q_bin_tensor, q_len_tensor,
            targets, n_rows
        )

    def numberize_sents_to_tensor(self, sents):
        sents_num = [self.vocab.numberize_sentence(s) for s in sents]

        return self.get_numberized_tensor(sents_num)

    def get_numberized_tensor(self, sents_num):
        """ sents_num: list of lists of word ids """
        dat_size = len(sents_num)
        bin_tensor = torch.LongTensor(dat_size, self.max_len)
        bin_tensor.fill_(vocab_pytorch.PAD_token)
        # slen_max = htable_params.max_len
        slen_tensor = torch.IntTensor(dat_size,)

        b = 0
        for sent_wordids in sents_num:
            slen = min(len(sent_wordids), self.max_len)
            slen_tensor[b] = slen
            for w in range(slen):
                wordid = sent_wordids[w]
                bin_tensor[b, slen - 1 - w] = wordid  # fill in reverse
            b += 1

        return sents_num, bin_tensor, slen_tensor
