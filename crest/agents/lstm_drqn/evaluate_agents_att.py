import logging
import numpy as np
from collections import namedtuple
import random
# from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F


from crest.helper.config_utils import change_config, get_prefix
from crest.helper.utils import read_file, save2file
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
    print('Loading DRQN agent')
    if config['general']['student']:
        agent = RLAgent(config, word_vocab, verb_map, noun_map, att=config['general']['use_attention'],
                        bootstrap=config['general']['student'], embed=config['bootstrap']['embed'])
    else:
        agent = RLAgent(config, word_vocab, verb_map, noun_map, att=config['general']['use_attention'], bootstrap=config        ['general']['student'],)
    return agent


class Evaluator():
    def __init__(self, config, args, threshold=0.3):
        self.config = config
        self.args = args
        teacher_path = config['general']['teacher_model_path']
        print('Setting up TextWorld environment...')

    def load_valid_env(self, valid_env_name):
        test_batch_size = 1
        valid_env_id = gym_textworld.make_batch(env_id=valid_env_name, batch_size=test_batch_size, parallel=True)
        self.valid_env = gym.make(valid_env_id)
        self.valid_env.seed(config['general']['random_seed'])
        print('Loaded env name: ', valid_env_name)
        
    def load_agent(self):
        self.agent = get_agent(config, self.valid_env)
        model_checkpoint_path = config['training']['scheduling']['model_checkpoint_path']
        load_path = model_checkpoint_path.replace('.pt', '_best.pt')
        print('Loading model from : ', load_path)
        self.agent.model.load_state_dict(torch.load(load_path))
        self.hidden_size = config['model']['lstm_dqn']['action_scorer_hidden_dim']
        self.hash_features = {}

    def inference(self, agent, env, prune=False, action_dist=None):
        batch_size = 1
        assert batch_size == 1, "Batchsize should be 1 during inference"
        agent.model.eval()
        obs, infos = env.reset()
        agent.reset(infos)
        id_string_0 = agent.get_observation_strings(infos)[0]

        provide_prev_action = self.config['general']['provide_prev_action']
        dones = [False] * batch_size

        rewards = [0]
        prev_actions = ["" for _ in range(batch_size)] if provide_prev_action else None

        input_description, description_id_list, desc, _ =\
                agent.get_game_step_info(obs, infos, prev_actions, prune=prune,
                                         teacher_actions=action_dist, ret_desc=True,)
        curr_ras_hidden, curr_ras_cell = None, None  # ras: recurrent action scorer

        if prune:
            desc_strings, desc_disc = agent.get_similarity_scores(obs, infos, prev_actions, prune=prune,
                                                                teacher_actions=action_dist, ret_desc=True,)
            self.desc.append(list(desc_disc.keys()))
            self.desc_scores.append(list(desc_disc.values()))
            self.desc_strings.append(desc_strings)
        
        if id_string_0 in self.id_string_list:
            print('Already encountered this game. Skipping...')
            return
        
        self.id_string_list.append(id_string_0)
        self.game_num += 1

        while not all(dones):
            v_idx, n_idx, _, curr_ras_hidden, curr_ras_cell = agent.generate_one_command(input_description, curr_ras_hidden, curr_ras_cell,
                                                                                         epsilon=0.0, return_att=args.use_attention)
            if args.use_attention:
                softmax_att = agent.get_softmax_attention()
            else:
                softmax_att = None
            qv, qn = agent.get_qvalues()
            _, v_idx_maxq, _, n_idx_maxq = agent.choose_maxQ_command(qv, qn)
            chosen_strings = agent.get_chosen_strings(v_idx_maxq.detach(), n_idx_maxq.detach())
            
            sorted_tokens=None
            sorted_atts=None
        
            obs, rewards, dones, infos = env.step(chosen_strings)
            if provide_prev_action:
                prev_actions = chosen_strings
            IR = [info["intermediate_reward"] for info in infos]
            if type(dones) is bool:
                dones = [dones] * batch_size

            agent.rewards.append(rewards)
            agent.dones.append(dones)
            agent.intermediate_rewards.append([info["intermediate_reward"] for info in infos])
            
            input_description, description_id_list, desc, _ =\
                agent.get_game_step_info(obs, infos, prev_actions, prune=prune, teacher_actions=action_dist, ret_desc=True,)
            
            if prune:
                desc_strings, desc_disc = agent.get_similarity_scores(obs, infos, prev_actions, prune=prune, teacher_actions=action_dist, ret_desc=True)
                self.desc.append(list(desc_disc.keys()))
                self.desc_scores.append(list(desc_disc.values()))
                self.desc_strings.append(desc_strings)

                _, _, orig_desc, _ = agent.get_game_step_info(obs, infos, prev_actions, prune=False, ret_desc=True,)
                for x, y in zip(orig_desc, desc):
                    self.orig_data += [' '.join(x), ' '.join(y)]
                
        agent.finish()
        R = agent.final_rewards.mean()
        S = agent.step_used_before_done.mean()
        IR = agent.final_intermediate_rewards.mean()

        msg = '====EVAL==== R={:.3f}, IR={:.3f}, S={:.3f}'
        msg = msg.format(R, IR, S)
        print(msg)
        print("\n")

        self.result_logs['R'].append(R)
        self.result_logs['IR'].append(IR)
        self.result_logs['S'].append(S)

    def infer(self):
        numgames = self.args.num_test_games
        prune = self.args.prune
        save_dict = {}
        count = 0
        self.id_string_list = []
        self.result_logs = {'R': [], 'IR': [], 'S': []}

        if prune:
            self.prune_filename  = config['training']['scheduling']['model_checkpoint_path'].replace('.pt', '_level_{}_logs.txt'.format(args.level)).replace('saved_models', 'prune_logs')
            self.score_filename  = config['training']['scheduling']['model_checkpoint_path'].replace('.pt', '_level_{}_logs.npz'.format(args.level)).replace('saved_models', 'score_logs')
            self.orig_data = []
            self.desc = []
            self.desc_scores = []
            self.desc_strings = []

        if args.method=='drqn':
            if prune:
                prefix_name = get_prefix(self.args)
                filename = './data/teacher_data/{}.npz'.format(prefix_name)
                teacher_dict = np.load(filename, allow_pickle=True)
                global_action_set = set()
                for k in teacher_dict.keys():
                    if k=='allow_pickle':
                        continue
                    action_dist = teacher_dict[k][-1]
                    action_dist = [x for x in action_dist.keys()]
                    global_action_set.update(action_dist)
            self.game_num = 0
            print('here')
            while (len(self.id_string_list)<int(numgames)):
                print('Game number : ', self.game_num)
                if prune:
                    self.inference(self.agent, self.valid_env, prune=prune, action_dist=[list(global_action_set)])
                else:
                    self.inference(self.agent, self.valid_env, prune=False)
        if prune:
            save2file(self.prune_filename, self.orig_data)
            np.savez(self.score_filename, desc=self.desc, desc_scores=self.desc_scores, desc_strings=self.desc_strings)
        return self.result_logs

if __name__ == '__main__':
    import os, argparse, pickle
    parser = argparse.ArgumentParser(description="train network.")
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    parser.add_argument("-type", "--type", default=None, help="easy | medium | hard")
    parser.add_argument("-ng", "--num_games", default=None, help="easy | medium | hard")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-vv", "--very-verbose", help="print out warnings", action="store_true")
    parser.add_argument("-fr", "--force-remove", help="remove experiments directory to start new", action="store_true")
    parser.add_argument("-att", "--use_attention", help="Use attention in the encoder model", action="store_true")
    parser.add_argument("-th", "--threshold", help="Filter threshold value for cosine similarity", default=0.3, type=float)
    parser.add_argument("-ea", "--exp_act", help="Use expanded vocab list for actions", action="store_true")
    parser.add_argument("-prune", "--prune", help="Use pruning or not", action="store_true")
    parser.add_argument("-level", "--level", help="how many levels in the game to test", type=int, default=15)
    parser.add_argument("-m", "--method", help="What method to use DRQN/DQN", type=str, default="drqn")
    parser.add_argument("-emb", "--embed", default='cnet', type=str) # 'cnet' | 'glove' | 'word2vec' | 'bert'
    parser.add_argument("-drop", "--dropout", default=0, type=float)
    parser.add_argument("-student", "--student", help="Whether Teacher or Student model", action="store_true")
    args = parser.parse_args()
    assert not args.force_remove

    config = change_config(args, method=args.method, test=True)

    args.num_test_games = 20

    true_valid_name = config['general']['valid_env_id']
    evaluator = Evaluator(config, args, threshold=args.threshold)
    evaluator.load_valid_env(true_valid_name)
    evaluator.load_agent()

    filename  = config['training']['scheduling']['model_checkpoint_path'].replace('.pt', '_level_{}_logs.txt'.format(args.level)).replace('saved_models', 'emnlp_logs/logs_{}_{}'.format(args.type, args.num_games))

    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    fp = open(filename, 'w')

    results = []
    for k in range(3):
        config['general']['valid_env_id'] = true_valid_name
        config['general']['valid_env_id'] = config['general']['valid_env_id'].replace('gamesize10', 'gamesize20')
        config['general']['valid_env_id'] = config['general']['valid_env_id'].replace('_validation', '_test')
        config['general']['valid_env_id'] = config['general']['valid_env_id'].replace('_step50', '_step100').replace('_step75', '_step100')
        config['general']['valid_env_id'] = config['general']['valid_env_id'].replace('_level15', '_level{}'.format(args.level))
        config['general']['valid_env_id'] = config['general']['valid_env_id'].replace('_seed9', '_seed{}'.format(k+1))
        fp.writelines('##'*30)
        fp.writelines('\n')
        fp.writelines(config['general']['valid_env_id'])
        fp.writelines('\n')
        fp.writelines('##'*30)
        fp.writelines('\n')
        
        evaluator.load_valid_env(config['general']['valid_env_id'])
        result_logs = evaluator.infer()

        R = np.mean(result_logs['R'])
        IR = np.mean(result_logs['IR'])
        S = np.mean(result_logs['S'])
        results.append([R, IR, S])
        msg = '====FINAL EVAL==== R={:.3f}, IR={:.3f}, S={:.3f}'.format(R, IR, S)
        fp.writelines(msg)
        fp.writelines('\n')

    mean_res = np.mean(results, axis=0)
    std_res = np.std(results, axis=0)
    fp.writelines('##' * 30)
    fp.writelines('\n')
    fp.writelines(' Final seeded results : R={}/{}, IR={}/{}, S={}/{}'.format(mean_res[0], std_res[0], mean_res[1], std_res[1], mean_res[2], std_res[2]))
    fp.writelines('\n')
    fp.writelines('##' * 30)
    fp.writelines('\n')
    fp.close()
    pid = os.getpid()
    os.system('kill -9 {}'.format(pid))
