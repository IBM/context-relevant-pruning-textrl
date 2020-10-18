PYTHON=$(which python)
# python -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type easy -ng 25 -att -student -fr -th 0.5 -prune"
if [ $1 = "bootstrap_easy" ]; then
    screen -S "easy_25" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type easy -ng 25 -att -student -fr -th 0.5 -prune"
    screen -S "easy_50" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type easy -ng 50 -att -student -fr -th 0.5 -prune"
    screen -S "easy_500" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type easy -ng 500 -att -student -fr -th 0.5 -prune"
fi

if [ $1 = "bootstrap_medium" ]; then
    screen -S "medium_50" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type medium -ng 50 -att -student -fr -th 0.7 -prune"
    screen -S "medium_100" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type medium -ng 100 -att -student -fr -th 0.7 -prune"
    screen -S "medium_500" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type medium -ng 500 -att -student -fr -th 0.7 -prune"
fi

if [ $1 = "bootstrap_hard" ]; then
    screen -S "hard_50" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type hard -ng 50 -att -student -fr -th 0.7 -prune"
    screen -S "hard_100" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_policy_qlearn -c config -type hard -ng 100 -att -student -fr -th 0.7 -prune"
fi

if [ $1 = "base_drqn_easy" ]; then
    screen -S "base_easy_25" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type easy -ng 25 -att -fr"
    screen -S "base_easy_50" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type easy -ng 50 -att -fr "
    screen -S "base_easy_500" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type easy -ng 500 -att -fr"
fi

if [ $1 = "base_drqn_medium" ]; then
    screen -S "base_medium_50" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type medium -ng 50 -att -fr"
    screen -S "base_medium_100" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type medium -ng 100 -att -fr"
    screen -S "base_medium_500" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type medium -ng 500 -att -fr"
fi

if [ $1 = "base_drqn_hard" ]; then
    screen -S "base_hard_50" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type hard -ng 50 -att -fr"
    screen -S "base_hard_100" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type hard -ng 100 -att -fr"
    # screen -S "base_hard_500" -dm bash -c "${PYTHON} -m crest.agents.lstm_drqn.train_single_generate_agent -c config -type hard -ng 500 -att -fr"
fi