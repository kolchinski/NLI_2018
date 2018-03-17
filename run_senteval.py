import sys
sys.path.append('SentEval/examples/')
# sys.path.append('./src/models')

# PATH_SENTEVAL = '../'
# sys.path.insert(0, PATH_SENTEVAL)
sys.path.append('SentEval/')
import senteval
# import SentEval.senteval as senteval

import torch.nn as nn
import numpy as np
import src.dataman.wrangle as wrangle
import src.models.load_embeddings as load_embeddings
from src.models.seq2seq_model_pytorch import Seq2SeqPytorch
import src.models.model_pipeline_pytorch as model_pipeline_pytorch
import src.models.siamese_pytorch as siamese_pytorch
from src.utils import dotdict
import src.constants as constants

import torch
import torch.optim as optim
import sys
import logging
import time
from pytorch_classification.utils import (
    Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig)



args = dotdict({
    'type': 'siamese',
    'encoder_type': 'rnn',
    'lr': 0.05,
    'use_dot_attention': True,
    'learning_rate_decay': 0.9,
    'max_length': 100,
    'epochs': 10,
    'batch_size': 64,
    'batches_per_epoch': 3000,
    'test_batches_per_epoch': 500,
    'input_size': 300,
    'hidden_size': 2048,
    'n_layers': 1,
    'bidirectional': False,
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
    'kfold': 10
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
        if True:
            model = siamese_pytorch.SiameseClassifier(config=args)
            model.embed.weight.data = load_embeddings.load_embeddings(
                dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)
        else:
            model = Seq2SeqPytorch(args=args, vocab=dm.vocab)
            model.encoder.embedding.weight.data = load_embeddings.\
                load_embeddings(
                    dm.vocab, constants.EMBED_DATA_PATH, args.embedding_size)

        model_pipeline_pytorch.load_checkpoint(model, checkpoint=checkpoint)

        sent_model = siamese_pytorch.SiameseClassifierSentEmbed(
            config=args, embed=model.embed, encoder=model.encoder)

        model.eval()
        if args.cuda:
            model = model.cuda()

        params['config'] = args
        params['dm'] = dm
        params['sent_model'] = sent_model

    def batcher(params, batch):
        ''' input batch is list of sentences (list of words/tokenized)'''
        config = params['config']
        dm = params['dm']
        model = params['sent_model']

        # numberize
        sents_num, sent_bin_tensor, sent_len_tensor = dm.\
            numberize_sents_to_tensor(batch)

        # prepare input data
        if config.encoder_type == 'transformer':
            sent, sent_posembinput = None  # TODO
            sent_unsort = None
            encoder_init_hidden = None
        elif config.encoder_type == 'rnn':
            sent, sent_unsort = None  # TODO
            sent_posembinput = None
            encoder_init_hidden = model.encoder.initHidden(
                batch_size=args.batch_size)

        embeddings = model(
            encoder_init_hidden=encoder_init_hidden,
            encoder_input=sent,
            encoder_pos_emb_input=sent_posembinput,
            encoder_unsort=sent_unsort,
            batch_size=config.batch_size
        )

        return embeddings

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
    results = se.eval(transfer_tasks)
    print(results)
