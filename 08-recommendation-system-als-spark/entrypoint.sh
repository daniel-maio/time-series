#!/bin/bash

SPARK_WORKLOAD=$1

echo "SPARK_WORKLOAD: $SPARK_WORKLOAD"

if [ "$SPARK_WORKLOAD" = "master" ]; then
  exec start-master.sh -p 7077

elif [ "$SPARK_WORKLOAD" = "worker" ]; then
  exec start-worker.sh $SPARK_MASTER

elif [ "$SPARK_WORKLOAD" = "history" ]; then
  exec start-history-server.sh

else
  echo "Invalid SPARK_WORKLOAD: $SPARK_WORKLOAD"
  exit 1
fi