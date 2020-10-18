
if [ "$1" = "medium" ]; then
  echo "###############################################################"
  echo "###########    Saving coin-collector medium data   ############"
  echo "###############################################################"
  python -m crest.agents.lstm_drqn.prepare_gist -c config -type medium -ng 50 -th 0.7 -att
  python -m crest.agents.lstm_drqn.prepare_gist -c config -type medium -ng 100 -th 0.7 -att
  python -m crest.agents.lstm_drqn.prepare_gist -c config -type medium -ng 500 -th 0.7 -att
fi

if [ "$1" = "easy" ]; then
  echo "###############################################################"
  echo "###########    Saving coin-collector easy data  ###############"
  echo "###############################################################"
  python -m crest.agents.lstm_drqn.prepare_gist -c config -type easy -ng 25 -th 0.5 -att
  python -m crest.agents.lstm_drqn.prepare_gist -c config -type easy -ng 50 -th 0.5 -att
  python -m crest.agents.lstms_drqn.prepare_gist -c config -type easy -ng 500 -th 0.5 -att
fi

if [ "$1" = "hard" ]; then
  echo "###############################################################"
  echo "###########    Saving coin-collector hard data  ###############"
  echo "###############################################################"
  python -m crest.agents.lstm_drqn.prepare_gist -c config -type hard -ng 50 -th 0.7 -att
  python -m crest.agents.lstm_drqn.prepare_gist -c config -type hard -ng 100 -th 0.7 -att
  python -m crest.agents.lstm_drqn.prepare_gist -c config -type hard -ng 500 -th 0.7 -att
fi


