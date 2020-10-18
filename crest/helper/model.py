import logging
import numpy as np
import torch
import torch.nn.functional as F
from crest.helper.layers import Embedding, masked_mean, LSTMCell, FastUniLSTM
import math
import torch.nn as nn

logger = logging.getLogger(__name__)

class LSTM_DQN(torch.nn.Module):
    model_name = 'lstm_dqn'

    def __init__(self, model_config, word_vocab, verb_map, noun_map, enable_cuda=False):
        super(LSTM_DQN, self).__init__()
        self.model_config = model_config
        self.enable_cuda = enable_cuda
        self.word_vocab_size = len(word_vocab)
        self.id2word = word_vocab
        self.n_actions = len(verb_map)
        self.n_objects = len(noun_map)
        self.read_config()
        self._def_layers()
        self.init_weights()
        # self.print_parameters()

    def print_parameters(self):
        amount = 0
        for p in self.parameters():
            amount += np.prod(p.size())
        print("total number of parameters: %s" % (amount))
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        amount = 0
        for p in parameters:
            amount += np.prod(p.size())
        print("number of trainable parameters: %s" % (amount))

    def read_config(self):
        # model config
        config = self.model_config[self.model_name]
        self.embedding_size = config['embedding_size']
        self.encoder_rnn_hidden_size = config['encoder_rnn_hidden_size']
        self.action_scorer_hidden_dim = config['action_scorer_hidden_dim']
        # import ipdb; ipdb.set_trace()
        self.dropout_between_rnn_layers = config['dropout_between_rnn_layers']

    def _def_layers(self):
        # word embeddings
        self.word_embedding = Embedding(embedding_size=self.embedding_size, vocab_size=self.word_vocab_size, enable_cuda=self.enable_cuda)

        # lstm encoder
        self.encoder = FastUniLSTM(ninp=self.embedding_size, nhids=self.encoder_rnn_hidden_size,
                                   dropout_between_rnn_layers=self.dropout_between_rnn_layers)

        # Recurrent network for temporal dependencies (a.k.a history).
        self.action_scorer_shared_recurrent = LSTMCell(input_size=self.encoder_rnn_hidden_size[-1],
                                                       hidden_size=self.action_scorer_hidden_dim)

        self.action_scorer_shared = torch.nn.Linear(self.encoder_rnn_hidden_size[-1], self.action_scorer_hidden_dim)
        self.action_scorer_action = torch.nn.Linear(self.action_scorer_hidden_dim, self.n_actions, bias=False)
        self.action_scorer_object = torch.nn.Linear(self.action_scorer_hidden_dim, self.n_objects, bias=False)
        self.fake_recurrent_mask = None

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.action_scorer_shared.weight.data, gain=1)
        torch.nn.init.xavier_uniform_(self.action_scorer_action.weight.data, gain=1)
        torch.nn.init.xavier_uniform_(self.action_scorer_object.weight.data, gain=1)
        self.action_scorer_shared.bias.data.fill_(0)

    def representation_generator(self, _input_words):
        embeddings, mask = self.word_embedding.forward(_input_words)  # batch x time x emb
        encoding_sequence, _, _ = self.encoder.forward(embeddings, mask)  # batch x time x h
        mean_encoding = masked_mean(encoding_sequence, mask)  # batch x h
        return mean_encoding

    def recurrent_action_scorer(self, state_representation, last_hidden=None, last_cell=None):
        # state representation: batch x input
        # last hidden / last cell: batch x hid
        if self.fake_recurrent_mask is None or self.fake_recurrent_mask.size(0) != state_representation.size(0):
            self.fake_recurrent_mask = torch.autograd.Variable(torch.ones(state_representation.size(0),))
            if self.enable_cuda:
                self.fake_recurrent_mask = self.fake_recurrent_mask.cuda()

        new_h, new_c = self.action_scorer_shared_recurrent.forward(state_representation, self.fake_recurrent_mask,
                                                                   last_hidden, last_cell)
        action_rank = self.action_scorer_action.forward(new_h)  # batch x n_action
        object_rank = self.action_scorer_object.forward(new_h)  # batch x n_object
        return action_rank, object_rank, new_h, new_c

    def action_scorer(self, state_representation):
        hidden = self.action_scorer_shared.forward(state_representation)  # batch x hid
        hidden = F.relu(hidden)  # batch x hid
        action_rank = self.action_scorer_action.forward(hidden)  # batch x n_action
        object_rank = self.action_scorer_object.forward(hidden)  # batch x n_object
        return action_rank, object_rank

from torch.autograd import Variable
class LSTM_DQN_ATT(LSTM_DQN):
    model_name = 'lstm_dqn'

    def __init__(self, *args, **kwargs):
        super(LSTM_DQN_ATT, self).__init__(*args, **kwargs)
        # self.attn = torch.nn.Linear(self.encoder_rnn_hidden_size[0], 1)
        self.attn_inner = torch.nn.Linear(self.encoder_rnn_hidden_size[0], 32)
        self.attn_outer = torch.nn.Linear(32, 1, bias=False)
        if self.enable_cuda:
            self.attn_inner = self.attn_inner.cuda()
            self.attn_outer = self.attn_outer.cuda()

    def representation_generator(self, _input_words, return_att=False, att_mask=None):
        embeddings, mask = self.word_embedding.forward(_input_words)  # batch x time x emb
        encoding_sequence, _, _ = self.encoder.forward(embeddings, mask)  # batch x time x h

        softmax_att = torch.zeros(encoding_sequence.shape[:-1], requires_grad=True)
        if self.enable_cuda:
            softmax_att = softmax_att.cuda()
        for i in range(len(encoding_sequence)):
            numel = int(torch.sum(mask[i]).item())
            logit_attn = self.attn_outer(F.tanh(self.attn_inner(encoding_sequence[i][:numel])))
            softmax_att[i, :numel] = F.softmax(logit_attn, 0).squeeze(-1)
        
        if att_mask is not None:
            softmax_att = softmax_att * att_mask
        mean_encoding = torch.bmm(softmax_att.unsqueeze(1), encoding_sequence).squeeze(1)

        if return_att:
            return mean_encoding, softmax_att
        else:
            return mean_encoding