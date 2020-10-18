# 1- gametype
# 2- numgames
# dqn no att

PYTHON=$(which python)


# # drqn att
${PYTHON} -m textrl.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -level 15 -att -level 20

if [ $1 = "easy" ]; then
    # drqn att
    ${PYTHON} -m textrl.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.5 -prune -level 20
fi

if [ $1 = "medium" ]; then
    ${PYTHON} -m textrl.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.7 -prune -level 20
fi

if [ $1 = "hard" ]; then
    ${PYTHON} -m textrl.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.7 -prune -level 20
fi

# drqn att
# ${PYTHON} -m textrl.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -level 15 -att -level 25

if [ $1 = "easy" ]; then
    # drqn att
    ${PYTHON} -m textrl.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.5 -prune -level 25
fi

if [ $1 = "medium" ]; then
    ${PYTHON} -m textrl.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.7 -prune -level 25
fi

if [ $1 = "hard" ]; then
    ${PYTHON} -m textrl.agents.lstm_drqn.evaluate_agents_att -c config -type ${1} -ng ${2} -att -student -th 0.7 -prune -level 25
fi



