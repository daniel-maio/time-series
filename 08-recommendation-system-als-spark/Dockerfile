FROM python:3.11-bullseye

ARG SPARK_VERSION=3.5.6

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      sudo \
      curl \
      vim \
      nano \
      unzip \
      rsync \
      openjdk-11-jdk \
      build-essential \
      software-properties-common \
      ssh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV SPARK_HOME=/opt/spark
ENV HADOOP_HOME=/opt/hadoop

RUN mkdir -p ${SPARK_HOME} ${HADOOP_HOME}

WORKDIR ${SPARK_HOME}

RUN curl https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz \
    -o spark.tgz && \
    tar xvzf spark.tgz --directory ${SPARK_HOME} --strip-components 1 && \
    rm spark.tgz

ENV PATH="${SPARK_HOME}/sbin:${SPARK_HOME}/bin:${PATH}"

ENV SPARK_MASTER="spark://spark-master:7077"
ENV SPARK_MASTER_HOST=spark-master
ENV SPARK_MASTER_PORT=7077
ENV PYSPARK_PYTHON=python3
ENV PYTHONPATH="${SPARK_HOME}/python/:${PYTHONPATH}"

COPY config/spark-defaults.conf ${SPARK_HOME}/conf/
COPY entrypoint.sh .

RUN chmod u+x ${SPARK_HOME}/sbin/* && \
    chmod u+x ${SPARK_HOME}/bin/* && \
    chmod +x entrypoint.sh

RUN mkdir -p /app

WORKDIR /app

ENTRYPOINT ["/opt/spark/entrypoint.sh"]