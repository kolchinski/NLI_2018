# adapted from https://github.com/libowen2121/SNLI-decomposable-attention
# original author: Bowen Li

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class encoder(nn.Module):

    def __init__(self, num_embeddings, embedding_size, hidden_size, para_init, fix_emb, intra_attn):
        super(encoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.para_init = para_init
        self.intra_attn = intra_attn

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        if fix_emb:
            self.embedding.weight.requires_grad = False

        self.input_linear = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        if self.intra_attn:
            self.f_intra = self._mlp_layers(self.hidden_size, self.hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                if m.bias is not None:
                    m.bias.data.normal_(0, self.para_init)

    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent):
        '''
               sent: batch_size x length (Long tensor)
        '''
        batch_size = sent.size(0)
        len = sent.size(1)

        sent = self.embedding(sent)

        sent = sent.view(-1, self.embedding_size)

        sent_linear = self.input_linear(sent).view(
            batch_size, -1, self.hidden_size)

        # sent_linear: batch_size x length x hidden_size
        if self.intra_attn:
            f = self.f_intra(sent_linear.view(-1, self.hidden_size))
            f = f.view(-1, len, self.hidden_size)
            # batch_size x length x hidden_size
            score = torch.bmm(f, torch.transpose(f, 1, 2))
            # f_{ij} batch_size x len x len
            prob = F.softmax(score.view(-1, len), dim=1).view(-1, len, len)
            # batch_size x len x len
            sent_attn = torch.bmm(prob, sent_linear)
            # batch_size x length x hidden_size
            return torch.cat((sent_linear,sent_attn),2)

        else:
            return sent_linear

class DecomposableClassifierSentEmbed(nn.Module):
    def __init__(self, config, embed, encoder):
        super(DecomposableClassifierSentEmbed, self).__init__()
        self.config = config
        self.encoder = encoder

    def forward(
        self,
        encoder_input,
        encoder_len,
        encoder_pos_emb_input,
        encoder_unsort,
        encoder_init_hidden,
        batch_size,
    ):
        sent_words_embed = self.encoder(sent = encoder_input) #batch_size x len x hidden_sizeh

        len = sent_words_embed.size(1)
        mask = Variable(torch.zeros(sent_words_embed.size()))
        mask = mask.byte()
        if self.config.cuda:
            mask = mask.cuda()
        for i, _ in enumerate(encoder_len.data):
            l = encoder_len.data[i]
            if l < len:
                mask[l:, i, :] = 1
        sent_words_embed[mask] = 0

        # sent_embed = torch.max(sent_words_masked,1)[0] #batch_size x 1 x hidden_size
        sent_embed = torch.sum(sent_words_embed,1)
        sent_embed = torch.squeeze(sent_embed, 1) #batch_size x hidden_size
        return sent_embed

class SNLIClassifier(nn.Module):
    '''
        intra sentence attention
    '''

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()

        self.hidden_size = config.hidden_size
        self.label_size = config.d_out
        self.para_init = config.para_init
        self.intra_attn = config.intra_attn

        self.encoder = encoder(config.n_embed, config.embedding_size, self.hidden_size, self.para_init, config.fix_emb, config.intra_attn)

        if self.intra_attn:
            self.encoding_size = 2*self.hidden_size
        else:
            self.encoding_size = self.hidden_size

        self.mlp_f = self._mlp_layers(self.encoding_size, self.hidden_size, self.hidden_size)
        self.mlp_g = self._mlp_layers(2 * self.encoding_size, self.hidden_size, self.hidden_size)
        self.mlp_h = self._mlp_layers(2 * self.hidden_size, self.hidden_size, self.hidden_size)

        self.final_linear = nn.Linear(
            self.hidden_size, self.label_size, bias=True)

        self.log_prob = nn.LogSoftmax(dim=1)

        '''initialize parameters'''
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                if m.bias is not None:
                    m.bias.data.normal_(0, self.para_init)

    def _mlp_layers(self, input_dim, output_dim, hidden_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, hidden_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            hidden_dim, output_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    def forward(self, sent1, sent2):
        '''
                       sent: batch_size x length (Long tensor)
        '''

        # sent_linear: batch_size x length x encoding_size
        sent1_linear = self.encoder(sent=sent1)
        sent2_linear = self.encoder(sent=sent2)

        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)

        '''attend'''

        f1 = self.mlp_f(sent1_linear.view(-1, self.encoding_size))
        f2 = self.mlp_f(sent2_linear.view(-1, self.encoding_size))


        f1 = f1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size


        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = F.softmax(score1.view(-1, len2), dim=1).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous()
        # e_{ji} batch_size x len2 x len1
        prob2 = F.softmax(score2.view(-1, len1), dim=1).view(-1, len2, len1)
        # batch_size x len2 x len1

        sent1_combine = torch.cat(
            (sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (encoding_size x 2)
        sent2_combine = torch.cat(
            (sent2_linear, torch.bmm(prob2, sent1_linear)), 2)
        # batch_size x len2 x (encoding_size x 2)

        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.encoding_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.encoding_size))
        g1 = g1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        # if sample_id == 15:
        #     print '-2 layer'
        #     print h.data[:, 100:150]

        h = self.final_linear(h)

        # print 'final layer'
        # print h.data

        log_prob = self.log_prob(h)
        #log_prob = F.log_softmax(h, dim=1)

        return log_prob