#!/usr/bin/env python
"""Create setup script for pip installation"""
########################################################################
# File: setup.py
#  executable: setup.py
#
# Author: Andrew Bailey
# History: 06/27/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
from setuptools import setup
import os
from pip.req import parse_requirements



def main():
    """Main docstring"""
    start = timer()
    # pkg_name = 'nanotensor'
    # pkg_path = os.path.join(os.path.dirname(__file__), pkg_name)


    setup(
        name="nanotensor",
        version='0.2',
        description='BLSTM for basecaling ONT reads',
        url='https://github.com/BD2KGenomics/nanopore-RNN',
        author='Andrew Bailey',
        author_email='andbaile@ucsc.com',
        packages=['nanotensor'],
        install_requires=["biopython==1.69",
                          "boto==2.46.1",
                          "numpy>=1.12.1",
                          "pip>=9.0.1",
                          "pysam>=0.8.2.1",
                          "h5py>=2.6.0",
                          "python-dateutil==2.6.0",
                          "codecov==2.0.9",
                          "coverage==4.4.1",
                          "pytest==3.0.7"],
        zip_safe=False
    )


    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
