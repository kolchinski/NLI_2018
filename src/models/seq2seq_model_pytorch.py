# python 3
from models.basemodel import BaseModel
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from torch.autograd import Variable
import torch.nn.functional as F
import transformer_pytorch

use_cuda = torch.cuda.is_available()


class RNNEncoder(nn.Module):

    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config
        input_size = config.embedding_size
        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=config.hidden_size,
            num_layers=config.n_layers, dropout=config.dp_ratio,
            bidirectional=config.bidirectional)

    def initHidden(self, batch_size):
        if self.config.bidirectional:
            state_shape = 2, batch_size, self.config.hidden_size
        else:
            state_shape = 1, batch_size, self.config.hidden_size
        h0 = c0 = Variable(torch.zeros(state_shape))
        return (h0, c0)

    def forward(self, inputs, hidden, batch_size):
        outputs, (ht, ct) = self.rnn(inputs, hidden)
        return outputs


class TransformerEncoder(nn.Module):

    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config


class Seq2SeqPytorch(BaseModel):
    def __init__(self, args, vocab):
        BaseModel.__init__(self)
        self.lr = args.lr
        self.max_length = args.max_length
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.batches_per_epoch = args.batches_per_epoch
        self.test_batches_per_epoch = args.test_batches_per_epoch
        self.embedding_size = args.embedding_size
        self.output_size = 3
        self.cuda = args.cuda

        self.encoder_hidden_size = args.hidden_size
        self.decoder_hidden_size = args.hidden_size

        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.encoder = EncoderRNN(
            hidden_size=self.encoder_hidden_size,
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
        )
        self.decoder = DecoderRNN(
            hidden_size=self.decoder_hidden_size,
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            output_size=self.output_size,
        )

        self.criterion = nn.NLLLoss()

        self.net = Seq2Seq(encoder=self.encoder, decoder=self.decoder)


class Seq2Seq(nn.Module):

    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.embedding_size)
        if self.config.fix_emb:
            self.embed.weight.requires_grad = False

        if config.encoder_type == 'transformer':
            self.encoder = transformer_pytorch.Encoder(
                n_src_vocab=config.n_embed,
                n_max_seq=config.max_length,
                src_word_emb=self.embed,
            )
            self.decoder = transformer_pytorch.Decoder(
                n_src_vocab=config.n_embed,
                n_max_seq=config.max_length,
                tgt_word_emb=self.embed,
            )
        else:
            raise("encoder_type not supported {}".format(config.encoder_type))

        # self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        self.tgt_word_proj = nn.Linear(config.d_model, 3, bias=False)

        seq_in_size = 4 * config.hidden_size
        if self.config.bidirectional:
            seq_in_size *= 2
        assert len(config.mlp_classif_hidden_size_list) == 2
        self.out = nn.Sequential(
            nn.Linear(seq_in_size, config.mlp_classif_hidden_size_list[0]),
            self.relu,
            nn.Linear(config.mlp_classif_hidden_size_list[0], config.mlp_classif_hidden_size_list[1]),
            self.relu,
            nn.Linear(config.mlp_classif_hidden_size_list[1], config.d_out),
        )

    def forward(
        self,
        encoder_input,
        encoder_pos_emb_input,
        encoder_unsort,
        decoder_input,
        decoder_pos_emb_input,
        decoder_unsort,
        encoder_init_hidden,
        batch_size
    ):
        if self.config.encoder_type == 'transformer':
            encoder_output = self.encoder(
                src_seq=encoder_input,
                src_pos=encoder_pos_emb_input,
            )
            decoder_output = self.decoder(
                tgt_seq=decoder_input,
                tgt_pos=decoder_pos_emb_input,
                src_seq=encoder_input,
                enc_output=encoder_output,
            )
            decoder_output_project = self.tgt_word_proj(decoder_output)
        else:
            raise('encoder_type not supported {}'.format(self.config.encoder_type))

        softmax_outputs = F.log_softmax(decoder_output_project, dim=0)  # [batch_size, 3]

        return softmax_outputs



class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
        )
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
        )

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_size))


class DecoderRNN(EncoderRNN):
    def __init__(self, hidden_size, vocab_size, embedding_size,
                 output_size):
        EncoderRNN.__init__(
            self, hidden_size=hidden_size,
            vocab_size=vocab_size, embedding_size=embedding_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = EncoderRNN.forward(self, input=input, hidden=hidden)
        last_output = self.out(hidden[-1])
        return F.log_softmax(last_output, dim=0)
