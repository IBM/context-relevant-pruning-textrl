#!/usr/bin/env bash

######################################################
############    Train/validation games  ##############
######################################################

###### Easy ######
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize10_step50_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize25_step50_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize50_step50_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize500_step50_seed9_train

python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize10_step50_seed9_validation
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize20_step50_seed9_validation

###### Medium ######
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize25_step75_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize50_step75_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize100_step75_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize500_step75_seed9_train

python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize10_step75_seed9_validation
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize20_step75_seed9_validation


###### Hard ######
python -m crest.generator.gym_textworld.scripts.tw-make twcc_hard_level15_gamesize25_step75_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_hard_level15_gamesize50_step75_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_hard_level15_gamesize100_step75_seed9_train
python -m crest.generator.gym_textworld.scripts.tw-make twcc_hard_level15_gamesize500_step75_seed9_train

python -m crest.generator.gym_textworld.scripts.tw-make twcc_hard_level15_gamesize10_step75_seed9_validation
python -m crest.generator.gym_textworld.scripts.tw-make twcc_hard_level15_gamesize20_step75_seed9_validation

##########################################
############    Test games  ##############
##########################################

###### Easy test sets ######
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize20_step100_seed1_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize20_step100_seed2_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level15_gamesize20_step100_seed3_test

python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level20_gamesize20_step100_seed1_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level20_gamesize20_step100_seed2_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level20_gamesize20_step100_seed3_test

python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level25_gamesize20_step100_seed1_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level25_gamesize20_step100_seed2_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_easy_level25_gamesize20_step100_seed3_test

###### Medium test sets ######
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize20_step100_seed1_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize20_step100_seed2_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level15_gamesize20_step100_seed3_test

python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level20_gamesize20_step100_seed1_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level20_gamesize20_step100_seed2_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level20_gamesize20_step100_seed3_test

python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level25_gamesize20_step100_seed1_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level25_gamesize20_step100_seed2_test
python -m crest.generator.gym_textworld.scripts.tw-make twcc_medium_level25_gamesize20_step100_seed3_test


