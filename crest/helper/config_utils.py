import warnings
import yaml
from os.path import join as pjoin
import textworld
import shutil
import time, os

def change_config(args, method='drqn', wait_time=2, kind='normal', tf=False, ensemble=False, test=False): # kind = noisy | normal
    student = args.student
    exp_act_list = args.exp_act
    challenge_type = 'coin_collector'

    if args.very_verbose:
        args.verbose = args.very_verbose
        warnings.simplefilter("default", textworld.TextworldGenerationWarning)

    # Read config from yaml file.
    if challenge_type == 'custom_tw' or challenge_type == 'treasure_hunter':
        config_file = pjoin(args.config_dir, 'config_{}_{}.yaml'.format(args.type, args.num_games))
    elif challenge_type == 'coin_collector':  # args.type is not None:
        if args.num_games is not None:
            config_file = pjoin(args.config_dir, 'config_{}_{}.yaml'.format(args.type, args.num_games))
        else:
            config_file = pjoin(args.config_dir, 'config_{}.yaml'.format(args.type, ))
    else:
        config_file = pjoin(args.config_dir, 'config.yaml')

    with open(config_file) as reader:
        config = yaml.safe_load(reader)

    prefixed_method_name = method + ('_att' if args.use_attention else '')
    prefixed_method_name = 'coin_collector_' + prefixed_method_name
        
    config['bootstrap']['threshold'] = args.threshold
    config['bootstrap']['prune'] = args.prune if hasattr(args, 'prune') else False
    config['bootstrap']['embed'] = args.embed if hasattr(args, 'embed') else 'cnet'
    use_dropout = args.dropout if hasattr(args, 'dropout') else None
    
    if exp_act_list:
        prefixed_method_name += '_exp_act'

    if use_dropout is not None and not student: # student model does not use drop-out
        prefixed_method_name += '_drop_{}'.format(use_dropout)

    ######## Base model path #################
    teacher_model_path = config['training']['scheduling']['model_checkpoint_path']. \
        replace('dqrn', prefixed_method_name). \
        replace('summary_', '').replace('.pt', '_train.pt')

    ######## Bootstrapped model path #################
    config['training']['scheduling']['teacher_model_checkpoint_path'] = \
        config['training']['scheduling']['model_checkpoint_path']. \
            replace('dqrn', prefixed_method_name). \
            replace('summary_', '')

    config['general']['student'] = student
    if student:
        print('##')
        prefixed_method_name += '_student'
         
        prefixed_method_name += '_thres_{}'.format(config['bootstrap']['threshold'])
        if config['bootstrap']['prune']:
            prefixed_method_name += '_prune'

        if config['bootstrap']['embed'] is not 'cnet':
            prefixed_method_name += '_embed_{}'.format(config['bootstrap']['embed'])

    ## Change the method specific info here
    config['general']['experiment_tag'] = config['general']['experiment_tag'].replace('drqn', method)
    config['general']['experiments_dir'] = config['general']['experiments_dir'].replace('summary', prefixed_method_name)

    config['training']['scheduling']['model_checkpoint_path'] = \
        config['training']['scheduling']['model_checkpoint_path']. \
            replace('dqrn', prefixed_method_name). \
            replace('summary_', '')

    config['general']['teacher_model_path'] = teacher_model_path

    config['general']['use_attention'] = args.use_attention
    config['general']['student'] = student
    config['general']['exp_act'] = exp_act_list

    print(config['general']['experiment_tag'])
    print(config['general']['experiments_dir'])
    print(config['training']['scheduling']['model_checkpoint_path'])

    print('Train env name : ', config['general']['env_id'])
    print('Valid env name : ', config['general']['valid_env_id'])

    time.sleep(wait_time)
    if os.path.exists(config['general']['experiments_dir']) and not test:
        if not args.force_remove: 
            prompt = input('Are you sure you want to delete {} (yes/no):'.
                        format(config['general']['experiments_dir']))
            # if prompt == 'yes' or do_not_prompt:
            if prompt == 'yes':
                print('##' * 30)
                print('Removing directory ', config['general']['experiments_dir'])
                print('##' * 30)
                shutil.rmtree(config['general']['experiments_dir'])
        else:
            print('##' * 30)
            print('Removing directory ', config['general']['experiments_dir'])
            print('##' * 30)
            shutil.rmtree(config['general']['experiments_dir'])
    else:
        if os.path.exists(config['general']['experiments_dir']):
            print('{} already exists. If you want to delete and '
                  'start fresh use \'-fr\' option.'.
                  format(config['general']['experiments_dir']))
    return config


def get_prefix(args, method='drqn'):
    prefixed_method_name = 'coin_collector_'
    prefixed_method_name += (method + ('_att' if args.use_attention else ''))
    if args.exp_act:
        prefixed_method_name += '_exp_act'
    prefixed_method_name += '_ng_{}'.format(args.num_games)
    prefixed_method_name += '_type_{}'.format(args.type)
    return prefixed_method_name