FROM gcc:5.4
FROM opentable/anaconda

MAINTAINER Andrew Bailey andbaile@ucsc.edu



RUN apt-get update && apt-get install -y \
    build-essential \
    git



WORKDIR /docker
COPY bwa-0.7.12.tar.bz2 /docker
RUN tar -xvjf bwa-0.7.12.tar.bz2
WORKDIR /docker/bwa-0.7.12
ENV PATH ${PATH}:/docker/bwa-0.7.12
RUN apt-get install -y make zlib1g-dev g++

RUN make


# RUN apt-get install -y python-pip
# RUN apt-get install -y cython
# RUN apt-get install -y python-numpy
# # RUN apt-get install -y marginAlign
# RUN apt-get install -y python-setuptools
# RUN apt-get install -y python-dev
# RUN apt-get install -y libhdf5-dev
# RUN apt-get install -y python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
# RUN apt-get install libz-dev

# RUN git clone --recursive -b toil_refactor https://github.com/ArtRand/signalAlign.git
WORKDIR /docker/signalAlign
WORKDIR /docker
ENV PYTHONPATH ${PYTHONPATH}:/docker/signalAlign


RUN git clone --recursive -b embed_labels https://github.com/adbailey4/signalAlign.git



WORKDIR /docker/signalAlign
RUN make

# RUN pip install h5py
# RUN apt-get install -y pysam
# RUN apt-get install -y pandas
# h5py==2.6.0
# numpy==1.11.2
# pandas==0.19.1
# pysam==0.8.2.1
# python-dateutil==2.6.0
# pytz==2016.10
# setuptools==32.1.0
# six==1.10.0
# wheel==0.29.0
# installs requirements and puts deps in path
WORKDIR /docker/signalAlign
RUN pip install -e .
# still necessary,
ENV PYTHONPATH ${PYTHONPATH}:/docker/signalAlign

ENV PATH ${PATH}:/docker/signalAlign/bin
# run c library and alignment tests
# numpy==1.9.2",
#                      "marginAlign==1.1.9",
#                      "h5py==2.6.0",
#                      "pysam==0.8.2.1",
#                      "pandas==0.18.1",
#                      "sonLib==1.1.0"]

COPY wrapper.sh /opt/signalAlign/

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["sh", "/opt/signalAlign/wrapper.sh"]
