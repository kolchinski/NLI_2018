import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import re
from collections import Counter
import nltk

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
sent_tokenize = lambda para: [para]

# somewhat adapted from https://github.com/allenai/bi-att-flow/blob/master/squad/prepro.py
def sample_squad_classif_dataset(gl_vocab, source_data, output_path, start_ratio=0.0, stop_ratio=1.0, MinAnswerLen=5):
    q = []
    a = []
    start_ai = int(round(len(source_data['data']) * start_ratio))
    stop_ai = int(round(len(source_data['data']) * stop_ratio))
    f = open(output_path, 'w')
    for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
        article_questions, article_answers = [], []
        for pi, para in enumerate(article['paragraphs']):
            for qa in para['qas']:
                qa_answers = []
                qi_tokenized = word_tokenize(qa['question'].lower())
                unk_ct = 0
                for word in qi_tokenized:
                    if word not in gl_vocab:
                        unk_ct += 1
                if unk_ct:
                    continue
                qi = ' '.join(qi_tokenized)
                for answer in qa['answers']:
                    unk_ct = 0
                    answer_tokenized = word_tokenize(answer['text'].lower())
                    for word in answer_tokenized:
                        if word not in gl_vocab:
                            unk_ct += 1
                    if unk_ct:
                        continue
                    answer_text = ' '.join(answer_tokenized)
                    qa_answers.append(answer_text)
                if not qa_answers:
                    continue
                longest_answer = max(qa_answers, key=len)
                if len(longest_answer.split()) >= MinAnswerLen \
                    and qi not in article_questions and longest_answer not in article_answers:  # dedupe
                    article_questions.append(qi)
                    article_answers.append(longest_answer)

        for i, a1 in enumerate(article_answers):
            for a2 in article_answers:
                if a1 not in a2 and a2 not in a1:  # not equal and not subset
                    if np.random.rand() < 0.5:
                        x1, x2 = a2, a1
                        yi = 1
                    else:
                        x1, x2 = a1, a2
                        yi = 0
                    qi = article_questions[i]
                    f.write('{}\t{}\t{}\t{}\n'.format(yi, x1, x2, qi))
                    q.append(qi)
                    a.append((a1, a2))
    print('sampled {} examples'.format(len(q)))
    print('sampled unique questions {}'.format(len(set(q))))
    f.close()
    return q, a