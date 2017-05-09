# try to create docker image for nanonet


FROM adbailey4/nanonettrain

MAINTAINER Andrew Bailey andbaile@ucsc.edu



##### The following lines come from
# https://github.com/omsai/poretools/commit/0878bb366379469d57655229a19e68a5312535e0

############# BEGIN INSTALLATION ##############
ENV DEBIAN_FRONTEND noninteractive

# Prepare to install R
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
RUN echo 'deb http://cran.rstudio.com/bin/linux/ubuntu trusty/' >> /etc/apt/sources.list
RUN apt-get update

 # Install dependencies
RUN apt-get -y install git python-tables python-setuptools python-pip python-dev cython libhdf5-serial-dev r-base python-rpy2 libfreetype6-dev pkg-config

# Upgrade numexpr
RUN easy_install --upgrade numpy
RUN pip install numexpr --upgrade

 # Install R packages
RUN Rscript -e 'options("repos" = c(CRAN = "http://cran.rstudio.com/")); install.packages("codetools"); install.packages("MASS"); install.packages("ggplot2")'

# Install poretools
RUN git clone https://github.com/arq5x/poretools /docker/poretools
WORKDIR /docker/poretools
RUN pip install -r requirements.txt
RUN pip install --upgrade six
RUN python setup.py install
############# END INSTALLATION ##############
################################


RUN apt-get install -y \
    build-essential \
    cmake \
    hdf5-tools \
    libhdf5-dev \
    hdf5-helpers \
    apt-utils\
    python-numpy\
    cython\
    wget\
    parallel
# install bwa
WORKDIR /docker
COPY bwa-0.7.12.tar.bz2 /docker
RUN tar -xvjf bwa-0.7.12.tar.bz2
WORKDIR /docker/bwa-0.7.12
RUN make
#
# # install samtools
# WORKDIR /docker
# RUN wget https://github.com/samtools/samtools/releases/download/1.3.1/samtools-1.3.1.tar.bz2
# RUN tar -xvjf samtools-1.3.1.tar.bz2
# WORKDIR /docker/samtools-1.3.1
# RUN make
# RUN mkdir /build
# RUN make prefix="/docker/samtools-1.3.1/build" install
RUN sudo apt-get install samtools

# install nanopolish
WORKDIR /docker
RUN git clone --recursive https://github.com/jts/nanopolish.git
WORKDIR /docker/nanopolish
# RUN git checkout e4403627c037945b118d0dfa72ef3fffcb1cbce8
RUN make


# install biopython and pysam
RUN pip install biopython
RUN pip install pysam


# install nanomod
# WORKDIR /docker
# RUN git clone https://github.com/scottgigante/nanomod.git
# WORKDIR /docker/nanomod
# RUN git checkout c25964a2bd966404d676235ff73557652aa91b91
# RUN python setup.py install


RUN mkdir /opt/nanomod
COPY wrapper.sh /opt/nanomod/

# RUN mkdir /data
WORKDIR mkdir /docker/nanomod
COPY nanomod /docker/nanomod
WORKDIR /docker/nanomod
RUN python setup.py install
# WORKDIR /data

# RUN git checkout e4403627c037945b118d0dfa72ef3fffcb1cbce8

ENV PATH ${PATH}:/docker/samtools-1.3.1/build/bin
ENV PATH ${PATH}:/docker/bwa-0.7.12
ENV PATH ${PATH}:/docker/nanopolish


ENTRYPOINT ["sh", "/opt/nanomod/wrapper.sh"]
