import numpy as np
from tqdm import tqdm
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def word_tokenize(tokens):
    return [
        token.replace("''", '"').replace("``", '"')
        for token in nltk.word_tokenize(tokens)]


# somewhat adapted from
# https://github.com/allenai/bi-att-flow/blob/master/squad/prepro.py

def sample_nli_classif_dataset(
    source_data, output_path,
    n_negative=1, sim_thres=0.4,
    start_ratio=0.0, stop_ratio=1.0,
    target_label='contradiction',
):
    """source_data: [label, premise, hypothesis] pd.DataFrame"""
    contradiction_mask = np.where(source_data[0] == target_label)[0]
    print(len(contradiction_mask))

    premise = source_data.iloc[contradiction_mask, 1].values
    correct = source_data.iloc[contradiction_mask, 2].values
    sampled_idx = np.random.permutation(n_negative * len(correct))

    tfidf = TfidfVectorizer()
    premise_tfidf = tfidf.fit_transform(premise)
    correct_tfidf = tfidf.transform(correct)

    sampled = []

    for i in range(len(premise)):
        prem = premise[i]
        true_hyp = correct[i]
        sim_prem_hyp = linear_kernel(premise_tfidf[i], correct_tfidf[i])
        if sim_prem_hyp > sim_thres:
            continue
        for j in range(n_negative):
            neg_idx = sampled_idx[i * n_negative + j] % len(correct)
            false_hyp = correct[neg_idx]
            if np.random.rand() < 0.5:
                hyp1, hyp2 = true_hyp, false_hyp
                label = 0
            else:
                hyp1, hyp2 = false_hyp, true_hyp
                label = 1
            sampled.append((label, prem, hyp1, hyp2))

    # shuffle
    perm = np.random.permutation(len(sampled))
    with open(output_path, 'w') as f:
        for i in range(len(sampled)):
            label, prem, hyp1, hyp2 = sampled[perm[i]]
            f.write('{}\t{}\t{}\t{}\n'.format(
                label, prem, hyp1, hyp2))

    print('wrote sampled to {}'.format(output_path))
