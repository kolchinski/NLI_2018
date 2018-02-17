from basemodel import BaseModel
import seq2seq_utils_pytorch
import seq2seq_train_pytorch
import torch.nn as nn


class Seq2SeqPytorch(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self)
        self.lr = args.lr
        self.max_length = args.max_length
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.batches_per_epoch = args.batches_per_epoch
        self.test_batches_per_epoch = args.test_batches_per_epoch
        self.input_size = args.input_size
        self.output_size = 3
        self.cuda = args.cuda

        self.encoder_hidden_size = args.hidden_size
        self.decoder_hidden_size = args.hidden_size

        self.encoder = seq2seq_utils_pytorch.EncoderRNN(
            input_size=self.input_size,
            hidden_size=self.encoder_hidden_size,
        )
        self.decoder = seq2seq_utils_pytorch.DecoderRNN(
            input_size=self.input_size,
            hidden_size=self.decoder_hidden_size,
            output_size=self.output_size)

        self.criterion = nn.NLLLoss()

        self.net = Seq2Seq(encoder=self.encoder, decoder=self.decoder)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(
            input_variable, input_lengths)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
