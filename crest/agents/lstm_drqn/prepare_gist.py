import logging
import numpy as np
from collections import namedtuple
import random
# from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append(sys.path[0] + "/..")

from crest.helper.config_utils import change_config, get_prefix
from crest.helper.utils import read_file
from crest.helper.bootstrap_utils import CREST
from crest.helper.nlp_utils import compact_text
from crest.helper.generic import dict2list
logger = logging.getLogger(__name__)

import gym
import gym_textworld  # Register all textworld environments.
from crest.agents.lstm_drqn.agent import RLAgent
from crest.agents.lstm_drqn.test_agent import test

def get_agent(config, env):
    word_vocab = dict2list(env.observation_space.id2w)
    word2id = {}
    for i, w in enumerate(word_vocab):
        word2id[w] = i

    if config['general']['exp_act']:
        print('##' * 30)
        print('Using expanded action list for treasure hunter')
        verb_list = read_file("data/vocabs/trial_run_custom_tw/verb_vocab.txt")
        object_name_list = read_file("data/vocabs/common_nouns.txt")
    else:
        verb_list = ["go", "take", "unlock", "lock", "drop", "look", "insert", "open", "inventory", "close"]
        object_name_list = ["east", "west", "north", "south", "coin", "apple", "carrot", "textbook", "passkey", "keycard"]

    # Add missing words in word2id
    for w in verb_list:
        if w not in word2id.keys():
            word2id[w] = len(word2id)
            word_vocab += [w, ]
    for w in object_name_list:
        if w not in word2id.keys():
            word2id[w] = len(word2id)
            word_vocab += [w, ]

    verb_map = [word2id[w] for w in verb_list if w in word2id]
    noun_map = [word2id[w] for w in object_name_list if w in word2id]
    print('Loaded {} verbs'.format(len(verb_map)))
    print('Loaded {} nouns'.format(len(noun_map)))
    print('##' * 30)

    print('Missing verbs and objects:')
    print([w for w in verb_list if w not in word2id])
    print([w for w in object_name_list if w not in word2id])

    agent = RLAgent(config, word_vocab, verb_map, noun_map, att=config['general']['use_attention'], bootstrap=False,)
    return agent


def topk_attention(softmax_att, desc, k=10):
    np_att = softmax_att.detach().cpu().numpy()[0]
    desc = desc[0]
    dtype = [('token', 'S10'), ('att', float)]
    values = [(s, a) for s, a in zip(desc, np_att)]
    val_array = np.array(values, dtype=dtype)
    sorted_values = np.sort(val_array, order='att')[::-1]

    sorted_tokens = [x['token'] for x in sorted_values]
    sorted_atts = [np.round(x['att'], 3) for x in sorted_values]
    return sorted_tokens[:k], sorted_atts[:k]


class GISTSaver():
    def __init__(self, config, args, threshold=0.3):
        self.bs_obj = CREST(threshold=threshold)
        self.config = config

        validation_games = 20

        teacher_path = config['general']['teacher_model_path']

        print('Setting up TextWorld environment...')
        self.batch_size = 1
        # load
        print('Making env id {}'.format(config['general']['env_id']))
        env_id = gym_textworld.make_batch(env_id=config['general']['env_id'],
                                          batch_size=self.batch_size,
                                          parallel=True)
        self.env = gym.make(env_id)
        # self.env.seed(config['general']['random_seed'])
    
        test_batch_size = config['training']['scheduling']['test_batch_size']

        # valid
        valid_env_name = config['general']['valid_env_id']
        valid_env_id = gym_textworld.make_batch(env_id=valid_env_name,
                                                batch_size=test_batch_size,
                                                parallel=True)
        self.valid_env = gym.make(valid_env_id)
        self.valid_env.seed(config['general']['random_seed'])

        self.teacher_agent = get_agent(config, self.env)
        print('Loading teacher from : ', teacher_path)
        self.teacher_agent.model.load_state_dict(torch.load(teacher_path))
        # import time; time.sleep(5)

        self.hidden_size = config['model']['lstm_dqn']['action_scorer_hidden_dim']
        self.hash_features = {}

    

    def inference_teacher(self, agent, env, noise_std=0):
        assert self.batch_size == 1, "Batchsize should be 1 during inference"
        agent.model.eval()
        obs, infos = env.reset()
        agent.reset(infos)
        id_string_0 = agent.get_observation_strings(infos)[0]
        print_command_string, print_rewards = [[] for _ in infos], [[] for _ in infos]
        print_interm_rewards = [[] for _ in infos]
        provide_prev_action = self.config['general']['provide_prev_action']
        dones = [False] * self.batch_size
        rewards = [0]
        prev_actions = ["" for _ in range(self.batch_size)] if provide_prev_action else None
        input_description, _, desc, _ = agent.get_game_step_info(obs, infos, prev_actions, ret_desc=True)
        curr_ras_hidden, curr_ras_cell = None, None  # ras: recurrent action scorer
        # curr_ras_hidden, curr_ras_cell = get_init_hidden(bsz=self.batch_size,
        #                                                  hidden_size=self.hidden_size, use_cuda=True)
        print("##" * 30)
        print(obs)
        print("##" * 30)
        obs_list = []
        infos_list = []
        act_list = []
        sorted_tokens_list = []
        sorted_att_list = []
        id_string = id_string_0
        new_rooms = 0
        while not all(dones):
            v_idx, n_idx, _, curr_ras_hidden, curr_ras_cell = agent.generate_one_command(input_description, curr_ras_hidden,
                                                                                         curr_ras_cell, epsilon=0.0, return_att=args.use_attention)
                                                                                         
            if args.use_attention:
                softmax_att = agent.get_softmax_attention()
            else:
                softmax_att = None
            qv, qn = agent.get_qvalues()
            qv_noisy = qv
            qn_noisy = qn

            _, v_idx_maxq, _, n_idx_maxq = agent.choose_maxQ_command(qv_noisy, qn_noisy)
            chosen_strings = agent.get_chosen_strings(v_idx_maxq.detach(), n_idx_maxq.detach())
            if args.use_attention:
                sorted_tokens, sorted_atts = topk_attention(softmax_att, desc, k=10)
            else:
                sorted_tokens = None
                sorted_atts = None
            print('Action : ', chosen_strings[0])

            obs_list.append(obs[0])
            infos_list.append(infos[0])
            act_list.append(chosen_strings[0])
            sorted_tokens_list.append(sorted_tokens)
            sorted_att_list.append(sorted_atts)

            obs, rewards, dones, infos = env.step(chosen_strings)
            if provide_prev_action:
                prev_actions = chosen_strings

            for i in range(len(infos)):
                print_command_string[i].append(chosen_strings[i])
                print_rewards[i].append(rewards[i])
                print_interm_rewards[i].append(infos[i]["intermediate_reward"])
            IR = [info["intermediate_reward"] for info in infos]

            new_id_string = agent.get_observation_strings(infos)[0]

            if new_id_string != id_string:
                self.hash_features[id_string] = [infos, prev_actions, qv.detach().cpu().numpy(),
                                                 qn.detach().cpu().numpy(),
                                                 softmax_att,
                                                 desc, chosen_strings]
                id_string = agent.get_observation_strings(infos)[0]
                new_rooms += 1
                if new_rooms >= 75:
                    break

            if type(dones) is bool:
                dones = [dones] * self.batch_size
            agent.rewards.append(rewards)
            agent.dones.append(dones)
            agent.intermediate_rewards.append([info["intermediate_reward"] for info in infos])

            input_description, _, desc, _ = agent.get_game_step_info(obs, infos, prev_actions, ret_desc=True)

        agent.finish()
        R = agent.final_rewards.mean()
        S = agent.step_used_before_done.mean()
        IR = agent.final_intermediate_rewards.mean()

        msg = '====EVAL==== R={:.3f}, IR={:.3f}, S={:.3f}, new_rooms={}'
        msg = msg.format(R, IR, S, new_rooms)
        print(msg)
        print("\n")
        return R, IR, S, obs_list, infos_list, act_list, sorted_tokens_list, \
               sorted_att_list, id_string_0

    def compute_similarity(self, state_, ):
        pass

    def compute_action_distribution(self, action_list, normalize=True):
        action_dict = {}
        tot_tokens = 0
        for action in action_list:
            for token in action.split(" "):
                tot_tokens += 1
                if token in action_dict.keys():
                    action_dict[token] += 1
                else:
                    action_dict[token] = 1
        if normalize:
            for token in action_dict.keys():
                action_dict[token] = (action_dict[token] * 1.)/tot_tokens
        return action_dict

    def infer(self, numgames, noise_std=0):
        save_dict = {}
        count = 0
        for i in range(numgames):
            print('Game number : ', i)
            R, IR, S, obs_list, infos_list, act_list, sorted_tokens_list, \
               sorted_att_list, id_string = \
                self.inference_teacher(self.teacher_agent, self.env, noise_std=noise_std)
            action_dist = self.compute_action_distribution(act_list)
            if R==1:
                count+=1
                save_dict[id_string] = [obs_list, infos_list, act_list,
                                        sorted_tokens_list, sorted_att_list,
                                        action_dist]
        print('saved ', count)
        prefix_name = get_prefix(args)
        filename = './data/teacher_data/{}.npz'.format(prefix_name)
        hash_filename = './data/teacher_data/teacher_softmax_{}.pkl'.format(prefix_name)
        np.savez(filename, **save_dict, allow_pickle=True)
        with open(hash_filename, 'wb') as fp:
             pickle.dump(self.hash_features, fp, -1)


if __name__ == '__main__':
    import os, argparse, pickle, hickle
    for _p in ['saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    parser = argparse.ArgumentParser(description="train network.")
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    parser.add_argument("-type", "--type", default=None, help="easy | medium | hard")
    parser.add_argument("-ng", "--num_games", default=25, type=int)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-vv", "--very-verbose", help="print out warnings", action="store_true")
    parser.add_argument("-fr", "--force-remove", help="remove experiments directory to start new",
                        action="store_true")
    parser.add_argument("-att", "--use_attention", help="Use attention in the encoder model",
                        action="store_true")
    parser.add_argument("-student", "--student", help="Use student", action="store_true")
    parser.add_argument("-th", "--threshold", help="Filter threshold value for cosine similarity", default=0.3, type=float)
    parser.add_argument("-ea", "--exp_act", help="Use expanded vocab list for actions", action="store_true")
    parser.add_argument("-drop", "--dropout", default=0, type=float)

    args = parser.parse_args()

    config = change_config(args, method='drqn', wait_time=0, test=True)
    
    state_pruner = GISTSaver(config, args, threshold=args.threshold)
    state_pruner.infer(args.num_games)

    pid = os.getpid()
    os.system('kill -9 {}'.format(pid))