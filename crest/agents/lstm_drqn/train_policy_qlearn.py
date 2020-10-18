import logging
import os
import numpy as np
import argparse
import warnings
import yaml
from os.path import join as pjoin
import sys
sys.path.append(sys.path[0] + "/..")
import pickle
import torch
from torch import nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from crest.agents.lstm_drqn.agent import RLAgent as Agent
from crest.helper.config_utils import change_config, get_prefix
from crest.helper.bootstrap_utils import get_init_hidden
from crest.helper.generic import SlidingAverage, to_np
from crest.helper.generic import get_experiment_dir, dict2list
from crest.agents.lstm_drqn.test_agent import test
from crest.helper.utils import read_file

logger = logging.getLogger(__name__)

import gym
import gym_textworld  # Register all textworld environments.

import textworld
# os.system('rm -r gen_games')


def train(config, prune=False, embed='cnet'):
    # train env
    print('Setting up TextWorld environment...')
    batch_size = config['training']['scheduling']['batch_size']
    env_id = gym_textworld.make_batch(env_id=config['general']['env_id'],
                                      batch_size=batch_size,
                                      parallel=True)
    env = gym.make(env_id)
    env.seed(config['general']['random_seed'])

    print("##" * 30)
    if prune:
        print('Using state pruning ...')
    else:
        print('Not using state pruning ...')
    print("##" * 30)
            
    # valid and test env
    run_test = config['general']['run_test']
    if run_test:
        test_batch_size = config['training']['scheduling']['test_batch_size']
        # valid
        valid_env_name = config['general']['valid_env_id']

        valid_env_id = gym_textworld.make_batch(env_id=valid_env_name, batch_size=test_batch_size, parallel=True)
        valid_env = gym.make(valid_env_id)
        valid_env.seed(config['general']['random_seed'])
        # valid_env.reset()

        # test
        test_env_name_list = config['general']['test_env_id']
        assert isinstance(test_env_name_list, list)

        test_env_id_list = [gym_textworld.make_batch(env_id=item, batch_size=test_batch_size, parallel=True) for item in test_env_name_list]
        test_env_list = [gym.make(test_env_id) for test_env_id in test_env_id_list]
        for i in range(len(test_env_list)):
            test_env_list[i].seed(config['general']['random_seed'])
            # test_env_list[i].reset()
    print('Done.')

    # Set the random seed manually for reproducibility.
    np.random.seed(config['general']['random_seed'])
    torch.manual_seed(config['general']['random_seed'])
    if torch.cuda.is_available():
        if not config['general']['use_cuda']:
            logger.warning("WARNING: CUDA device detected but 'use_cuda: false' found in config.yaml")
        else:
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(config['general']['random_seed'])
    else:
        config['general']['use_cuda'] = False  # Disable CUDA.
    use_cuda = config['general']['use_cuda']
    revisit_counting = config['general']['revisit_counting']
    replay_batch_size = config['general']['replay_batch_size']
    history_size = config['general']['history_size']
    update_from = config['general']['update_from']
    replay_memory_capacity = config['general']['replay_memory_capacity']
    replay_memory_priority_fraction = config['general']['replay_memory_priority_fraction']

    word_vocab = dict2list(env.observation_space.id2w)
    word2id = {}
    for i, w in enumerate(word_vocab):
        word2id[w] = i

    
    if config['general']['exp_act']:
        print('##' * 30)
        print('Using expanded verb list')
        verb_list = read_file("data/vocabs/trial_run_custom_tw/verb_vocab.txt")
        object_name_list = read_file("data/vocabs/common_nouns.txt")
    else:
        #"This option only works for coin collector"
        verb_list = ["go", "take", "unlock", "lock", "drop", "look", "insert", "open", "inventory", "close"]
        object_name_list = ["east", "west", "north", "south", "coin", "apple", "carrot", "textbook", "passkey",
                            "keycard"]
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
    
    # teacher_path = config['general']['teacher_model_path']
    # teacher_agent = Agent(config, word_vocab, verb_map, noun_map,
    #                         att=config['general']['use_attention'],
    #                         bootstrap=False,
    #                         replay_memory_capacity=replay_memory_capacity,
    #                         replay_memory_priority_fraction=replay_memory_priority_fraction)
    # teacher_agent.model.load_state_dict(torch.load(teacher_path))
    # teacher_agent.model.eval()
    
    student_agent = Agent(config, word_vocab, verb_map, noun_map,
                        att=config['general']['use_attention'],
                        bootstrap=config['general']['student'],
                        replay_memory_capacity=replay_memory_capacity,
                        replay_memory_priority_fraction=replay_memory_priority_fraction,
                        embed=embed)


    init_learning_rate = config['training']['optimizer']['learning_rate']
    exp_dir = get_experiment_dir(config)
    summary = SummaryWriter(exp_dir)

    parameters = filter(lambda p: p.requires_grad, student_agent.model.parameters())
    if config['training']['optimizer']['step_rule'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=init_learning_rate)
    elif config['training']['optimizer']['step_rule'] == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=init_learning_rate)
        
    log_every = 100
    reward_avg = SlidingAverage('reward avg', steps=log_every)
    step_avg = SlidingAverage('step avg', steps=log_every)
    loss_avg = SlidingAverage('loss avg', steps=log_every)

    # save & reload checkpoint only in 0th agent
    best_avg_reward = -10000
    best_avg_step = 10000

    # step penalty
    discount_gamma = config['general']['discount_gamma']
    provide_prev_action = config['general']['provide_prev_action']

    # epsilon greedy
    epsilon_anneal_epochs = config['general']['epsilon_anneal_epochs']
    epsilon_anneal_from = config['general']['epsilon_anneal_from']
    epsilon_anneal_to = config['general']['epsilon_anneal_to']

    # counting reward
    revisit_counting_lambda_anneal_epochs = config['general']['revisit_counting_lambda_anneal_epochs']
    revisit_counting_lambda_anneal_from = config['general']['revisit_counting_lambda_anneal_from']
    revisit_counting_lambda_anneal_to = config['general']['revisit_counting_lambda_anneal_to']
    model_checkpoint_path = config['training']['scheduling']['model_checkpoint_path']

    epsilon = epsilon_anneal_from
    revisit_counting_lambda = revisit_counting_lambda_anneal_from
    
    #######################################################################
    #####               Load the teacher data                         #####
    #######################################################################
    prefix_name = get_prefix(args)
    filename = './data/teacher_data/{}.npz'.format(prefix_name)
    teacher_dict = np.load(filename, allow_pickle=True)
    # import ipdb; ipdb.set_trace()
    global_action_set = set()

    print("##" * 30)
    print("Training for {} epochs".format(config['training']['scheduling']['epoch']))
    print("##" * 30)

    import time
    t0 = time.time()

    for epoch in range(config['training']['scheduling']['epoch']):
        student_agent.model.train()
        obs, infos = env.reset()
        student_agent.reset(infos)

        # this the string identifier for leading the episodic action distribution
        id_string = student_agent.get_observation_strings(infos)

        cont_flag=False
        for id_ in id_string:
            if id_ not in teacher_dict.keys():
                cont_flag=True

        if cont_flag:
            print('Skipping this epoch/.....')
            continue

        # Episodic action list
        action_dist = [teacher_dict[id_string[k]][-1] for k in range(len(id_string))]
        action_dist = [[x for x in item.keys()] for item in action_dist]

        for item in action_dist:
            global_action_set.update(item)

        print_command_string, print_rewards = [[] for _ in infos], [[] for _ in infos]
        print_interm_rewards = [[] for _ in infos]
        print_rc_rewards = [[] for _ in infos]
        dones = [False] * batch_size
        rewards = None
        avg_loss_in_this_game = []

        curr_observation_strings = student_agent.get_observation_strings(infos)
        if revisit_counting:
            student_agent.reset_binarized_counter(batch_size)
            revisit_counting_rewards = student_agent.get_binarized_count(curr_observation_strings)

        current_game_step = 0
        prev_actions = ["" for _ in range(batch_size)] if provide_prev_action else None

        input_description, description_id_list, student_desc, _ =\
            student_agent.get_game_step_info(obs, infos, prev_actions, prune=prune,
                                            teacher_actions=action_dist, ret_desc=True,)

        curr_ras_hidden, curr_ras_cell = None, None  # ras: recurrent action scorer
        memory_cache = [[] for _ in range(batch_size)]
        solved = [0 for _ in range(batch_size)]

        while not all(dones):
            student_agent.model.train()

            v_idx, n_idx, chosen_strings, curr_ras_hidden, curr_ras_cell = \
                student_agent.generate_one_command(input_description, curr_ras_hidden,
                                                curr_ras_cell, epsilon=0.0,
                                                return_att=args.use_attention)

            obs, rewards, dones, infos = env.step(chosen_strings)
            curr_observation_strings = student_agent.get_observation_strings(infos)
            # print(chosen_strings)
            if provide_prev_action:
                prev_actions = chosen_strings
            # counting
            if revisit_counting:
                revisit_counting_rewards = student_agent.get_binarized_count(curr_observation_strings, update=True)
            else:
                revisit_counting_rewards = [0.0 for b in range(batch_size)]
            student_agent.revisit_counting_rewards.append(revisit_counting_rewards)
            revisit_counting_rewards = [float(format(item, ".3f")) for item in revisit_counting_rewards]

            for i in range(len(infos)):
                print_command_string[i].append(chosen_strings[i])
                print_rewards[i].append(rewards[i])
                print_interm_rewards[i].append(infos[i]["intermediate_reward"])
                print_rc_rewards[i].append(revisit_counting_rewards[i])
            if type(dones) is bool:
                dones = [dones] * batch_size

            student_agent.rewards.append(rewards)
            student_agent.dones.append(dones)
            student_agent.intermediate_rewards.append([info["intermediate_reward"] for info in infos])

            # computer rewards, and push into replay memory
            rewards_np, rewards_pt, mask_np,\
            mask_pt, memory_mask = student_agent.compute_reward(revisit_counting_lambda=revisit_counting_lambda,
                                                                revisit_counting=revisit_counting)

            ###############################
            #####   Pruned state desc #####
            ###############################
            curr_description_id_list = description_id_list

            input_description, description_id_list, student_desc, _ =\
                student_agent.get_game_step_info(obs, infos, prev_actions, prune=prune,
                                                teacher_actions=action_dist, ret_desc=True,)

            for b in range(batch_size):
                if memory_mask[b] == 0:
                    continue
                if dones[b] == 1 and rewards[b] == 0:
                    # last possible step
                    is_final = True
                else:
                    is_final = mask_np[b] == 0
                if rewards[b] > 0.0:
                    solved[b] = 1
                # replay memory
                memory_cache[b].append(
                    (curr_description_id_list[b], v_idx[b], n_idx[b], rewards_pt[b], mask_pt[b], dones[b],
                     is_final, curr_observation_strings[b]))

            if current_game_step > 0 and current_game_step % config["general"]["update_per_k_game_steps"] == 0:
                policy_loss = student_agent.update(replay_batch_size, history_size, update_from, discount_gamma=discount_gamma)
                if policy_loss is None:
                    continue
                loss = policy_loss
                # Backpropagate
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(student_agent.model.parameters(), config['training']['optimizer']['clip_grad_norm'])
                optimizer.step()  # apply gradients
                avg_loss_in_this_game.append(to_np(policy_loss))
            current_game_step += 1

        for i, mc in enumerate(memory_cache):
            for item in mc:
                if replay_memory_priority_fraction == 0.0:
                    # vanilla replay memory
                    student_agent.replay_memory.push(*item)
                else:
                    # prioritized replay memory
                    student_agent.replay_memory.push(solved[i], *item)

        student_agent.finish()

        avg_loss_in_this_game = np.mean(avg_loss_in_this_game)
        reward_avg.add(student_agent.final_rewards.mean())
        step_avg.add(student_agent.step_used_before_done.mean())
        loss_avg.add(avg_loss_in_this_game)
        # annealing
        if epoch < epsilon_anneal_epochs:
            epsilon -= (epsilon_anneal_from - epsilon_anneal_to) / float(epsilon_anneal_epochs)
        if epoch < revisit_counting_lambda_anneal_epochs:
            revisit_counting_lambda -= (revisit_counting_lambda_anneal_from - revisit_counting_lambda_anneal_to) / float(revisit_counting_lambda_anneal_epochs)

        # Tensorboard logging #
        # (1) Log some numbers
        if (epoch + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            summary.add_scalar('avg_reward', reward_avg.value, epoch + 1)
            summary.add_scalar('curr_reward', student_agent.final_rewards.mean(), epoch + 1)
            summary.add_scalar('curr_interm_reward', student_agent.final_intermediate_rewards.mean(), epoch + 1)
            summary.add_scalar('curr_counting_reward', student_agent.final_counting_rewards.mean(), epoch + 1)
            summary.add_scalar('avg_step', step_avg.value, epoch + 1)
            summary.add_scalar('curr_step', student_agent.step_used_before_done.mean(), epoch + 1)
            summary.add_scalar('loss_avg', loss_avg.value, epoch + 1)
            summary.add_scalar('curr_loss', avg_loss_in_this_game, epoch + 1)
            t1 = time.time()
            summary.add_scalar('time', t1 - t0, epoch + 1)

        msg = 'E#{:03d}, R={:.3f}/{:.3f}/IR{:.3f}/CR{:.3f}, S={:.3f}/{:.3f}, L={:.3f}/{:.3f}, epsilon={:.4f}, lambda_counting={:.4f}'
        msg = msg.format(epoch,
                         np.mean(reward_avg.value), student_agent.final_rewards.mean(), student_agent.final_intermediate_rewards.mean(), student_agent.final_counting_rewards.mean(),
                         np.mean(step_avg.value), student_agent.step_used_before_done.mean(),
                         np.mean(loss_avg.value), avg_loss_in_this_game,
                         epsilon, revisit_counting_lambda)
        if (epoch + 1) % config["training"]["scheduling"]["logging_frequency"] == 0:
            torch.save(student_agent.model.state_dict(), model_checkpoint_path.replace('.pt', '_train.pt'))
            print("=========================================================")
            for prt_cmd, prt_rew, prt_int_rew, prt_rc_rew in zip(print_command_string, print_rewards, print_interm_rewards, print_rc_rewards):
                print("------------------------------")
                print(prt_cmd)
                print(prt_rew)
                print(prt_int_rew)
                print(prt_rc_rew)
        print(msg)
        # test on a different set of games
        if run_test and (epoch) % config["training"]["scheduling"]["logging_frequency"] == 0:
            valid_R, valid_IR, valid_S = test(config, valid_env, student_agent, test_batch_size, word2id, prune=prune,
                                              teacher_actions=[list(global_action_set)]*test_batch_size)
            summary.add_scalar('valid_reward', valid_R, epoch + 1)
            summary.add_scalar('valid_interm_reward', valid_IR, epoch + 1)
            summary.add_scalar('valid_step', valid_S, epoch + 1)

            # save & reload checkpoint by best valid performance
            if valid_R > best_avg_reward or (valid_R == best_avg_reward and valid_S < best_avg_step):
                best_avg_reward = valid_R
                best_avg_step = valid_S
                torch.save(student_agent.model.state_dict(), model_checkpoint_path.replace('.pt', '_best.pt'))
                print("========= saved checkpoint =========")


if __name__ == '__main__':
    for _p in ['saved_models']:
        if not os.path.exists(_p):
            os.mkdir(_p)
    parser = argparse.ArgumentParser(description="train network.")
    parser.add_argument("-c", "--config_dir", default='config', help="the default config directory")
    parser.add_argument("-type", "--type", default=None, help="easy | medium | hard")
    parser.add_argument("-ng", "--num_games", default=None, help="easy | medium | hard")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-vv", "--very-verbose", help="print out warnings", action="store_true")
    parser.add_argument("-fr", "--force-remove", help="remove experiments directory to start new", action="store_true")
    parser.add_argument("-att", "--use_attention", help="Use attention in the encoder model", action="store_true")
    parser.add_argument("-student", "--student", help="Whether Teacher or Student model", action="store_true")
    parser.add_argument("-th", "--threshold", help="Filter threshold value for cosine similarity", default=0.3, type=float)
    parser.add_argument("-ea", "--exp_act", help="Use expanded vocab list for actions", action="store_true")
    parser.add_argument("-prune", "--prune", help="Use pruning or not", action="store_true")
    parser.add_argument("-emb", "--embed", default='cnet', type=str) # 'cnet' | 'glove' | 'word2vec' | 'bert'
    args = parser.parse_args()
    
    config = change_config(args)
    
    print('Threshold: ', config['bootstrap']['threshold'])
    config['training']['scheduling']['epoch']=6000
    config['general']['epsilon_anneal_epochs']=3600
    config["training"]["scheduling"]["logging_frequency"] = 50

    train(config=config, prune=args.prune, embed=args.embed)
    
    # pid = os.getpid()
    # os.system('kill -9 {}'.format(pid))
    