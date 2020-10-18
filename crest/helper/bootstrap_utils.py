import string
import bcolz
import numpy as np
import os
import pickle

from .nlp_utils import compact_text
import nltk
import torch
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
stop_words = set(stopwords.words('english'))
eps = 10e-8

def get_init_hidden(bsz, hidden_size, use_cuda):
    h_0 = torch.autograd.Variable(torch.FloatTensor(bsz, hidden_size).zero_())
    c_0 = torch.autograd.Variable(torch.FloatTensor(bsz, hidden_size).zero_())

    if use_cuda:
        h_0, c_0 = h_0.cuda(), c_0.cuda()

    return h_0, c_0

def similarity(w1, w2, w2v):
    try:
        vec1 = w2v[w1]
    except KeyError:
        vec1 = np.zeros((300,))
    try:
        vec2 = w2v[w2]
    except KeyError:
        vec2 = np.zeros((300,))

    if np.sum(vec1)==0 and np.sum(vec2)!=0:
        vec1 = vec1[:len(vec2)]
    elif np.sum(vec1)!=0 and np.sum(vec2)==0:
        vec2 = vec2[:len(vec1)]

    unit_vec1 = vec1/(np.linalg.norm(vec1) + eps)
    unit_vec2 = vec2/(np.linalg.norm(vec2) + eps)
    return np.dot(unit_vec1, unit_vec2)


def normalize(state, remove_articles=True):
    state = state.lower()
    out = state.translate(str.maketrans('', '' , string.punctuation))
    if remove_articles:
        out = word_tokenize(out)
        s_ws = [w for w in out if not w in stop_words]
    else:
        s_ws = word_tokenize(out)
    return s_ws


def get_thresholded(sim_dict, t=0.2):
    final_words = []
    for k, v in sim_dict.items():
        if v >=t:
            final_words.append(k)
    return final_words


def statistics_score(sim_dict, kind='avg'):
    scores = []
    for k, v in sim_dict.items():
        scores.append(v)
    return np.mean(scores) if kind == 'avg' else np.max(scores)


def correlate_state(s_ws, object_list, w2v, mean=True):
    sim_dict = {}
    if isinstance(object_list, list):
        for w in s_ws:
            sim_list = []
            for w_obj in object_list:
                sim_list.append(similarity(w_obj, w, w2v))
            sim_dict[w] = np.mean(sim_list) if mean else np.max(sim_list)
    elif isinstance(object_list, dict):
        for w in s_ws:
            sim_list = []
            for w_obj, val in object_list.items():
                sim_list.append(val * similarity(w_obj, w, w2v))
            sim_dict[w] = np.mean(sim_list) if mean else np.max(sim_list)
    return sim_dict


class BootstrapFilter:
    def __init__(self, threshold=0.3,
                 filter_sentence=False):
        self.threshold = threshold
        self.load_cc_embeddings()
        self.load_bs_action_token()
        self.filter_sent = filter_sentence

    def load_cc_embeddings(self):
        embed_size = 300
        cc_path = os.path.expanduser('~/Data/nlp/conceptNet')
        filename = '{0}/numberbatch-en-19.08.txt'.format(cc_path)
        rootdir = '{0}/glove.dat'.format(cc_path)
        words_file = '{0}/CC_words.pkl'.format(cc_path, embed_size)
        idx_file = '{0}/CC_idx.pkl'.format(cc_path, embed_size)

        words = pickle.load(open(words_file, 'rb'))
        word2idx = pickle.load(open(idx_file, 'rb'))
        vectors = bcolz.open(rootdir)
        self.w2v = {w: vectors[word2idx[w]] for w in words}
    
# Domain relevant episodic state pruning
class CREST(BootstrapFilter):
    def __init__(self, threshold=0.3, embeddings='cnet'): # 'cnet' | 'glove' | 'word2vec' | 'bert'
        self.threshold = threshold

        print('##'*30)
        print('Using embedding : ', embeddings)
        print('##'*30)

        if embeddings=='cnet':
            self.load_cc_embeddings()
        elif embeddings=='glove':
            self.load_glove_embeddings()
        elif embeddings=='word2vec':
            self.load_w2v_embeddings()

    def load_glove_embeddings(self):
        embed_size=100
        glove_path=os.path.expanduser('~/Data/nlp/glove/glove.6B')
        rootdir = '{0}/6B.{1}.dat'.format(glove_path, embed_size)
        words_file = '{0}/6B.{1}_words.pkl'.format(glove_path, embed_size)
        idx_file = '{0}/6B.{1}_idx.pkl'.format(glove_path, embed_size)

        vectors = bcolz.open(rootdir)[:]
        words = pickle.load(open(words_file, 'rb'))
        word2idx = pickle.load(open(idx_file, 'rb'))
        self.w2v = {w: vectors[word2idx[w]] for w in words}

    def load_w2v_embeddings(self):
        self.w2v=gensim.models.KeyedVectors.load_word2vec_format('data/Googlemodel.bin',binary=True)

    def prune_state(self, noisy_string, expert_words, return_details=False, add_prefix=True):
        def get_scores(noisy_string_x, mean=True):
            s_wsx = normalize(noisy_string_x)
            sim_dictx = correlate_state(s_wsx, object_list=expert_words, w2v=self.w2v, mean=mean)
            return sim_dictx
        sentence_pruned_str_joined = noisy_string
        sim_dict = get_scores(sentence_pruned_str_joined, mean=False)
        final_str = get_thresholded(sim_dict, t=self.threshold)
        if add_prefix:
            final_str = '-= Unknown =- ' + ' '.join(final_str)
        else:
            final_str = ' '.join(final_str)
        if return_details:
            return final_str, sim_dict
        else:
            return final_str
