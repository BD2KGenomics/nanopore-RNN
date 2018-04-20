[![Build Status](https://travis-ci.org/UCSC-nanopore-cgl/nanopore-RNN.svg?branch=master)](https://travis-ci.org/UCSC-nanopore-cgl/nanopore-RNN)
[![codecov](https://codecov.io/gh/UCSC-nanopore-cgl/nanopore-RNN/branch/master/graph/badge.svg)](https://codecov.io/gh/UCSC-nanopore-cgl/nanopore-RNN)

# NanoTensor

We propose a series of scripts, which are still in development, to label and train a multilayer Bidirectional long short-term memory recurrent neural network to base-call ONT-nanopore reads with modified bases.



## INSTALLATION
Install BWA and make sure executable is in the PATH:
* `git clone https://github.com/lh3/bwa.git`
* `cd bwa`
* `make`
* `export PATH=$PATH:$PWD`

The easiest way to deal with the dependencies is to download Anaconda
* Download and Install Anaconda https://docs.continuum.io/anaconda/install
* `git clone --recursive https://github.com/BD2KGenomics/nanopore-RNN`
* `cd nanopore-RNN`
* `conda env create --file requirements.yml`
* `source activate nanotensor`
* `make`  
* C executables created by signalAlign need to be in the PATH
* `export PATH=$PATH:$PWD/signalAlign/bin`
* Test installation
* `make test`

## USAGE

##### TODO

#### Training using Nanonet

If you want to use Nanonet to train a network, you can use the script located [here](https://github.com/adbailey4/signalAlign/blob/embed_labels/src/signalalign/embed_signalalign.py) to embed aligned kmers into the fast5 file which can then be used with Nanonet.


### Contributions

If you decide to help contribute please use pylint so our code is consistent.
https://www.pylint.org/
