#!/bin/bash
USERNAME=ubuntu
if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  echo "Start script: runOnCloud.sh ip_of_machine"
  exit
fi

if [[ ! -f training_data.tar.gz ]]; then
  tar -zcvf training_data.tar.gz ./training_set
fi

if [[ ! -f validation_set.tar.gz ]]; then
  tar -zcvf validation_set.tar.gz test-set
fi

ssh $USERNAME@$1 'mkdir -p dog-vs-muffin'
scp validation_set.tar.gz $USERNAME@$1:./dog-vs-muffin/validation_set.tar.gz
scp training_data.tar.gz $USERNAME@$1:./dog-vs-muffin/training_data.tar.gz

ssh $USERNAME@$1 'tar -xf dog-vs-muffin/validation_set.tar.gz -C ./dog-vs-muffin/'
ssh $USERNAME@$1 'tar -xf dog-vs-muffin/training_data.tar.gz -C ./dog-vs-muffin/'
