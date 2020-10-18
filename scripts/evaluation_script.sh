# 1- gametype
# 2- numgames
PYTHON=$(which python)

# drqn no att
${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -level 15
# python -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type medium -ng 50 -att -fr -drop 0.5

# drqn att
${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -level 15 -att

if [ $1 = "easy" ]; then
    # drqn att: CNET
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -th 0.5 -prune -student

    # drqn att: W2V
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -th 0.5 -prune -student --embed word2vec

    # drqn att: glove
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -th 0.5 -prune -student --embed glove
fi

if [ $1 = "medium" ]; then
    # ours drqn att: CNET
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -student -th 0.7 -prune
    # ours drqn att: W2V
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.8 -prune --embed word2vec
    # ours drqn att: glove
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.8 -prune --embed glove
fi

if [ $1 = "hard" ]; then
    # ours drqn att: CNET
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -student -th 0.8 -prune
    # ours drqn att:  W2V
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.8 -prune --embed word2vec
    # ours drqn att: glove
    ${PYTHON} -m crest.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.8 -prune --embed glove
fi

