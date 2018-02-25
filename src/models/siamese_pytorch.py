# adapted from https://github.com/pytorch/examples/blob/master/snli/model.py
import torch
import torch.nn as nn
from torch.autograd import Variable


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.embedding_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.hidden_size,
                        num_layers=config.n_layers, dropout=config.dp_ratio,
                        bidirectional=config.bidirectional)

    def initHidden(self, batch_size):
        state_shape = 1, batch_size, self.config.hidden_size
        h0 = c0 = Variable(torch.zeros(state_shape))
        return (h0, c0)

    def forward(self, inputs, hidden, batch_size):
        outputs, (ht, ct) = self.rnn(inputs, hidden)
        return ht[-1] if not self.config.bidirectional else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.embedding_size)
        if config.projection:
            self.projection = Linear(config.embedding_size, config.d_proj)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        seq_in_size = 2*config.hidden_size
        if self.config.bidirectional:
            seq_in_size *= 2
        lin_config = [seq_in_size]*2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(seq_in_size, config.d_out))

    def forward(self, encoder_input, decoder_input,
                encoder_init_hidden, batch_size):
        # encoder_input, decoder_input should be packed sequences
        # (already embedded)
        # prem_embed = self.embed(encoder_input)
        # hypo_embed = self.embed(decoder_input)

        prem_embed = encoder_input
        hypo_embed = decoder_input

        # if self.config.fix_emb:
        #     prem_embed = Variable(encoder_input)
        #     hypo_embed = Variable(decoder_input)

        if self.config.projection:
            prem_embed = self.relu(self.projection(prem_embed))
            hypo_embed = self.relu(self.projection(hypo_embed))
        premise = self.encoder(
            inputs=prem_embed,
            hidden=encoder_init_hidden,
            batch_size=batch_size
        )
        hypothesis = self.encoder(
            inputs=hypo_embed,
            hidden=encoder_init_hidden,
            batch_size=batch_size
        )
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores
