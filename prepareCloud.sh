#!/bin/bash
#!/bin/bash
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

ssh ubuntu@$1 'mkdir -p dog-vs-muffin'
scp validation_set.tar.gz ubuntu@$1:./dog-vs-muffin/validation_set.tar.gz
scp training_data.tar.gz ubuntu@$1:./dog-vs-muffin/training_data.tar.gz

ssh ubuntu@$1 'tar -xf dog-vs-muffin/validation_set.tar.gz -C ./dog-vs-muffin/'
ssh ubuntu@$1 'tar -xf dog-vs-muffin/training_data.tar.gz -C ./dog-vs-muffin/'
