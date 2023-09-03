#!/bin/bash
USERNAME=ubuntu
if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  echo "Start script: runOnCloud.sh ip_of_machine"
  exit
fi

mvn clean package -P gpu

scp djl/target/java-djl-dog-vs-muffin-djl-1.0-SNAPSHOT-shaded.jar $USERNAME@$1:./dog-vs-muffin/java-djl-dog-vs-muffin-djl-1.0-SNAPSHOT-shaded.jar

# kill old process if there are any
ssh $USERNAME@$1 'killall -9 java; rm -rf dog-vs-muffin/models'

# node manager and gpu node exporter are running on port 9100 and 9835 on the remote machine, map them to local host so that we can always scrape the same ip (localhost)
ssh -L localhost:9101:localhost:9100 -L localhost:9835:localhost:9835 $USERNAME@$1 'cd dog-vs-muffin; /home/$USERNAME/.sdkman/candidates/java/17.0.3.6.1-amzn/bin/java -jar java-djl-dog-vs-muffin-djl-1.0-SNAPSHOT-shaded.jar learn'
