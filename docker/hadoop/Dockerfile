# Single-node (pseudo-distributed) Apache Hadoop 3.3.6: HDFS + YARN + MapReduce.
# Python 3 is installed because Hadoop Streaming tasks run hadoop/mapper.py and reducer.py.
FROM eclipse-temurin:11-jdk-jammy

ARG HADOOP_VERSION=3.3.6
ARG HADOOP_URL=https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz

ENV HADOOP_HOME=/opt/hadoop \
    HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop \
    HDFS_NAMENODE_USER=root \
    HDFS_DATANODE_USER=root \
    HDFS_SECONDARYNAMENODE_USER=root \
    YARN_RESOURCEMANAGER_USER=root \
    YARN_NODEMANAGER_USER=root
ENV PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 curl procps netcat-openbsd \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fSL "$HADOOP_URL" -o /tmp/hadoop.tgz \
 && tar -xzf /tmp/hadoop.tgz -C /opt \
 && mv /opt/hadoop-${HADOOP_VERSION} $HADOOP_HOME \
 && rm /tmp/hadoop.tgz \
 && rm -rf $HADOOP_HOME/share/doc \
 && echo "export JAVA_HOME=${JAVA_HOME}" >> $HADOOP_CONF_DIR/hadoop-env.sh

COPY core-site.xml hdfs-site.xml mapred-site.xml yarn-site.xml $HADOOP_CONF_DIR/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /hadoop/dfs/name /hadoop/dfs/data

# 9870 NameNode UI | 9000 HDFS RPC | 8088 YARN UI | 9864 DataNode UI | 19888 JobHistory UI
EXPOSE 9870 9000 8088 9864 19888
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=10 \
  CMD hdfs dfsadmin -safemode get | grep -q OFF || exit 1
ENTRYPOINT ["/entrypoint.sh"]
