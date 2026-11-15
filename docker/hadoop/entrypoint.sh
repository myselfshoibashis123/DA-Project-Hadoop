#!/usr/bin/env bash
# Starts all Hadoop daemons in one container and keeps it running.
set -e
if [ ! -d /hadoop/dfs/name/current ]; then
  echo "Formatting NameNode (first start)..."
  hdfs namenode -format -force -nonInteractive
fi
hdfs --daemon start namenode
hdfs --daemon start datanode
yarn --daemon start resourcemanager
yarn --daemon start nodemanager
mapred --daemon start historyserver

echo "Waiting for HDFS to leave safe mode..."
hdfs dfsadmin -safemode wait
hdfs dfs -mkdir -p /safety_analytics/raw/images /safety_analytics/raw/videos \
  /safety_analytics/processed/frames /safety_analytics/detections /safety_analytics/analytics \
  /tmp /user/root
hdfs dfs -chmod -R 1777 /tmp
echo "Hadoop is ready:  NameNode UI http://localhost:9870   YARN UI http://localhost:8088"
jps
# keep the container alive and show daemon logs
exec tail -F $HADOOP_HOME/logs/*.log
