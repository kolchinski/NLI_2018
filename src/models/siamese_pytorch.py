# adapted from https://github.com/pytorch/examples/blob/master/snli/model.py
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import transformer_pytorch


class RNNEncoder(nn.Module):

    def __init__(self, config):
        super(RNNEncoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.embedding_size
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


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.embedding_size)
        if self.config.fix_emb:
            self.embed.weight.requires_grad = False

        if config.encoder_type == 'transformer':
            self.encoder = transformer_pytorch.Encoder(
                n_src_vocab=config.n_embed, n_max_seq=config.max_length
            )
        else:
            self.encoder = RNNEncoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()

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
        prem_embed = encoder_input
        hypo_embed = decoder_input

        if self.config.encoder_type == 'transformer':
            premise = self.encoder(
                src_seq=encoder_input,
                src_pos=encoder_pos_emb_input,
            )
        else:
            premise = self.encoder(
                inputs=prem_embed,
                hidden=encoder_init_hidden,
                batch_size=batch_size
            )
            premise = nn.utils.rnn.pad_packed_sequence(premise)[0]
            premise = premise.index_select(1, encoder_unsort)
        if self.config.encoder_type == 'transformer':
            hypothesis = self.encoder(
                src_seq=decoder_input,
                src_pos=decoder_pos_emb_input,
            )
        else:
            hypothesis = self.encoder(
                inputs=hypo_embed,
                hidden=encoder_init_hidden,
                batch_size=batch_size
            )
            hypothesis = nn.utils.rnn.pad_packed_sequence(hypothesis)[0]
            hypothesis = hypothesis.index_select(1, decoder_unsort)

        premise_maxpool = torch.max(premise, 0)[0]
        hypothesis_maxpool = torch.max(hypothesis, 0)[0]

        scores = self.out(torch.cat([
            premise_maxpool,
            hypothesis_maxpool,
            torch.abs(premise_maxpool - hypothesis_maxpool),
            premise_maxpool * hypothesis_maxpool,
        ], 1))  # [batch_size, 3]

        softmax_outputs = F.log_softmax(scores, dim=0)  # [batch_size, 3]

        return softmax_outputs
