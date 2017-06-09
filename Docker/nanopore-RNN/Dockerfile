
FROM gcc:5.4
FROM opentable/anaconda

MAINTAINER Andrew Bailey andbaile@ucsc.edu


# python packages for nanopore-RNN
WORKDIR /docker
RUN conda install biopython
RUN git clone -b bailey --recursive https://github.com/BD2KGenomics/nanopore-RNN.git

ENV PATH ${PATH}:/docker/nanopore-RNN/nanonet
WORKDIR /docker/nanopore-RNN/nanonet
RUN python setup.py install
WORKDIR /docker/nanopore-RNN

# signalalign below
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
WORKDIR /docker/signalAlign
WORKDIR /docker
ENV PYTHONPATH ${PYTHONPATH}:/docker/signalAlign


RUN git clone --recursive -b embed_labels https://github.com/adbailey4/signalAlign.git



WORKDIR /docker/signalAlign
RUN make
WORKDIR /docker/signalAlign
RUN pip install -e .
# still necessary,
ENV PYTHONPATH ${PYTHONPATH}:/docker/signalAlign

ENV PATH ${PATH}:/docker/signalAlign/bin
COPY wrapper.sh /opt/signalAlign/

RUN mkdir /data
WORKDIR /data

ENTRYPOINT ["sh", "/opt/signalAlign/wrapper.sh"]
