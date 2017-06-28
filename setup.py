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


def main():
    """Main docstring"""
    start = timer()

    setup(
        name="nanotensor",
        version='0.1',
        description='The funniest joke in the world',
        url='https://github.com/BD2KGenomics/nanopore-RNN',
        author='Andrew Bailey',
        author_email='andbaile@ucsc.com',
        packages=['nanotensor'],
        install_requires=[
            'tensorflow',
              ],
        zip_safe=False
    )


    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)

if __name__ == "__main__":
    main()
    raise SystemExit
