import sys
sys.path.append('SentEval/examples/')
# sys.path.append('./src/models')

# PATH_SENTEVAL = '../'
# sys.path.insert(0, PATH_SENTEVAL)
sys.path.append('SentEval/')
import senteval
# import SentEval.senteval as senteval

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim

import numpy as np
import src.dataman.wrangle as wrangle
import src.models.load_embeddings as load_embeddings
from src.models.seq2seq_model_pytorch import Seq2SeqPytorch
import src.models.model_pipeline_pytorch as model_pipeline_pytorch
import src.models.siamese_pytorch as siamese_pytorch
import src.models.decomposable_pytorch as decomposable_pytorch
from src.utils import dotdict
import src.constants as constants

import sys
import logging


args = dotdict({
    #'type': 'decomposable',
    #'encoder_type': 'decomposable',
    'type': 'siamese',
    'sent_embed_type': 'mix',
    'encoder_type': 'rnn',
    'lr': 0.05,
    'use_dot_attention': True,
    'learning_rate_decay': 0.9,
    'max_length': 50,
    'epochs': 10,
    'batch_size': 128,
    'batches_per_epoch': 3000,
    'test_batches_per_epoch': 500,
    'input_size': 300,
    #'hidden_size': 200, #For decomposable model
    'para_init': 0.01,
    'intra_attn': True, # if we use intra_attention for decomposable model
    'hidden_size': 1024,
    'layer1_hidden_size': 1024,
    'n_layers': 2,
    'bidirectional': True,
    'embedding_size': 300,
    'fix_emb': True,
    'dp_ratio': 0.0,
    'd_out': 3,  # 3 classes
    'mlp_classif_hidden_size_list': [512, 512],
    'cuda': torch.cuda.is_available(),
})
state = {k: v for k, v in args.items()}

# define senteval params
params_senteval = {
    'task_path': './SentEval/data/senteval_data',
    'usepytorch': True,
    'kfold': 5,
}
params_senteval['classifier'] = {
    'nhid': 0,
    'optim': 'adam',
    'batch_size': 64,
    'tenacity': 5,
    'epoch_size': 4
}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    def prepare(params, samples):
        print(args)
        checkpoint = sys.argv[1]
        print('found checkpoint dir {}'.format(checkpoint))

        dm = wrangle.DataManager(args)
        args.n_embed = dm.vocab.n_words
        if args.type == 'siamese':
            model = siamese_pytorch.SiameseClassifier(config=args)
            model.embed.weight.data = load_embeddings.load_embeddings(
                dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)
        elif args.type == 'decomposable':
            model = decomposable_pytorch.SNLIClassifier(config=args)
            model.encoder.embedding.weight.data = load_embeddings.load_embeddings(
                dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)
        else:
            model = Seq2SeqPytorch(args=args, vocab=dm.vocab)
            model.encoder.embedding.weight.data = load_embeddings.\
                load_embeddings(
                    dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)

        model_pipeline_pytorch.load_checkpoint(model, checkpoint=checkpoint)

        if args.type == 'siamese':
            sent_model = siamese_pytorch.SiameseClassifierSentEmbed(
                config=args, embed=model.embed, encoder=model.encoder)
        elif args.type == 'decomposable':
            sent_model = decomposable_pytorch.DecomposableClassifierSentEmbed(
                config=args, embed=None, encoder=model.encoder)

        model.eval()
        sent_model.eval()
        if args.cuda:
            model = model.cuda()
            sent_model = sent_model.cuda()

        params['config'] = args
        params['dm'] = dm
        params['sent_model'] = sent_model

    def batcher(params, batch):
        ''' input batch is list of sentences (list of words/tokenized)'''
        config = params['config']
        dm = params['dm']
        model = params['sent_model']

        sents = [' '.join(sent) for sent in batch]
        batch_size = len(batch)
        config.batch_size = batch_size

        # numberize
        sent_num, sent_bin_tensor, sent_len_tensor = dm.\
            numberize_sents_to_tensor(sents)

        # prepare input data
        if config.encoder_type == 'transformer':
            sent_len_tensor = Variable(sent_len_tensor)
            sent_bin_tensor = Variable(sent_bin_tensor)
            sent_posembinput = Variable(dm.get_pos_embedinputinput(sent_num))
            sent_unsort = None
            encoder_init_hidden = None
        elif config.encoder_type == 'rnn':
            sent_bin_tensor, sent_unsort = dm.vocab.\
                get_packedseq_from_sent_batch(
                    seq_tensor=sent_bin_tensor,
                    seq_lengths=sent_len_tensor,
                    embed=model.embed,
                    use_cuda=config.cuda,
                )
            sent_posembinput = None
            sent_len_tensor = Variable(sent_len_tensor)
            encoder_init_hidden = model.encoder.initHidden(
                batch_size=batch_size)
        elif args.encoder_type == 'decomposable':
            sent_bin_tensor = Variable(sent_bin_tensor)
            sent_len_tensor = Variable(sent_len_tensor)
            sent_posembinput = None
            sent_unsort = None
            encoder_init_hidden = None
        else:
            raise Exception('encoder_type not supported {}'.format(
                args.encoder_type))
        if config.cuda:
            model = model.cuda()
            if config.encoder_type == 'transformer':
                sent_bin_tensor = sent_bin_tensor.cuda()
                sent_len_tensor = sent_len_tensor.cuda()
                sent_posembinput = sent_posembinput.cuda()
            if config.encoder_type == 'rnn':
                sent_len_tensor = sent_len_tensor.cuda()
                if len(encoder_init_hidden):
                    encoder_init_hidden = [
                        x.cuda() for x in encoder_init_hidden]
                else:
                    encoder_init_hidden = encoder_init_hidden.cuda()
            if args.encoder_type == 'decomposable':
                sent_bin_tensor = sent_bin_tensor.cuda()


        embeddings = model(
            encoder_init_hidden=encoder_init_hidden,
            encoder_input=sent_bin_tensor,
            encoder_len=sent_len_tensor,
            encoder_pos_emb_input=sent_posembinput,
            encoder_unsort=sent_unsort,
            batch_size=batch_size
        ).data.cpu().numpy()

        return embeddings

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
    results = se.eval(transfer_tasks)
    print(results)
