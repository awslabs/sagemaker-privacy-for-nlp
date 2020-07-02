FROM openkbs/jdk-mvn-py3

USER root
RUN apt-get update
RUN sudo apt-get update
# RUN apt-get install -y curl unzip python3 python3-setuptools python3-pip python-dev python3-dev python-psutil
RUN apt-get install -y build-essential

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir torch==1.5.0
RUN pip3 install torchtext==0.6.0
RUN pip3 install --no-cache-dir pyspark==2.4.5
RUN pip3 install annoy==1.17.0
RUN pip3 install tqdm==4.46.0
RUN pip3 install matplotlib==3.3.2
RUN pip3 install seaborn==0.11.0
RUN pip3 install scikit-learn==0.23.2

ENV PYTHONUNBUFFERED=TRUE

# http://blog.stuart.axelbrooke.com/python-3-on-spark-return-of-the-pythonhashseed
ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

# Point Spark at proper python binary
ENV PYSPARK_PYTHON=/usr/bin/python3
# Point Spark at proper python binary
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

# Note: These are copied over from the source directory in the build.sh script
COPY train.py /opt/ml/processing/code/

ENTRYPOINT ["python3"]
