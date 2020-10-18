import logging
import os
import numpy as np
import argparse
import warnings
import yaml
from os.path import join as pjoin

import torch
from crest.agents.lstm_drqn.agent import RLAgent
from crest.helper.generic import get_experiment_dir
logger = logging.getLogger(__name__)
import gym
import gym_textworld  # Register all textworld environments.
import textworld


def test(config, env, agent, batch_size, word2id, prune=False, teacher_actions=None):
    agent.model.eval()
    obs, infos = env.reset()
    agent.reset(infos)
    print_command_string, print_rewards = [[] for _ in infos], [[] for _ in infos]
    print_interm_rewards = [[] for _ in infos]

    provide_prev_action = config['general']['provide_prev_action']

    dones = [False] * batch_size
    rewards = None
    prev_actions = ["" for _ in range(batch_size)] if provide_prev_action else None

    if prune:
        input_description, description_id_list, desc, _ = \
            agent.get_game_step_info(obs, infos, prev_actions, prune=prune,
                                             teacher_actions=teacher_actions, ret_desc=True, )
    else:
        input_description, _ = agent.get_game_step_info(obs, infos, prev_actions)
    curr_ras_hidden, curr_ras_cell = None, None  # ras: recurrent action scorer

    while not all(dones):
        v_idx, n_idx, chosen_strings, curr_ras_hidden, curr_ras_cell = agent.generate_one_command(input_description, 
                                                                                                  curr_ras_hidden, 
                                                                                                  curr_ras_cell, 
                                                                                                  epsilon=0.0)
        obs, rewards, dones, infos = env.step(chosen_strings)
        if provide_prev_action:
            prev_actions = chosen_strings

        for i in range(len(infos)):
            print_command_string[i].append(chosen_strings[i])
            print_rewards[i].append(rewards[i])
            print_interm_rewards[i].append(infos[i]["intermediate_reward"])
        if type(dones) is bool:
            dones = [dones] * batch_size
        agent.rewards.append(rewards)
        agent.dones.append(dones)
        agent.intermediate_rewards.append([info["intermediate_reward"] for info in infos])

        if prune:
            input_description, description_id_list, desc, _ = \
                agent.get_game_step_info(obs, infos, prev_actions, prune=prune,
                                         teacher_actions=teacher_actions, ret_desc=True, )
        else:
            input_description, _ = agent.get_game_step_info(obs, infos, prev_actions)

    agent.finish()
    R = agent.final_rewards.mean()
    S = agent.step_used_before_done.mean()
    IR = agent.final_intermediate_rewards.mean()

    msg = '====EVAL==== R={:.3f}, IR={:.3f}, S={:.3f}'
    msg = msg.format(R, IR, S)
    print(msg)
    print("\n")
    return R, IR, S

