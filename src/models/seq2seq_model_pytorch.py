# python 3
from basemodel import BaseModel
import seq2seq_utils_pytorch
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


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
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_init_hidden = self.encoder.initHidden()
        encoder_outputs, encoder_hidden = self.encoder(
            encoder_input, encoder_init_hidden)
        result = self.decoder(decoder_input, encoder_hidden)
        return result


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

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            result.cuda()
        return result


class DecoderRNN(EncoderRNN):
    def __init__(self, hidden_size, vocab_size, embedding_size,
                 output_size):
        EncoderRNN.__init__(
            self, hidden_size=hidden_size,
            vocab_size=vocab_size, embedding_size=embedding_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = EncoderRNN.forward(self, input=input, hidden=hidden)
        print(output)
        last_output = self.out(hidden[-1])
        return F.log_softmax(last_output)
