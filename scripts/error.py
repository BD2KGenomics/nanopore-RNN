#!/usr/bin/env python
"""Home made error messages"""
########################################################################
# File: error.py
#  executable: error.py
# Purpose: Create some informative errors when needed
#
# Author: Andrew Bailey
# History: 05/17/17 Created
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer

class PathError(Exception):
    """Error created to report path errors when dealing with file searching"""
    def __init__(self, value):
        super(PathError, self).__init__()
        self.value = value
    def __str__(self):
        return repr(self.value)


class Usage(Exception):
    '''Signal Usage error

    Used to signal a Usage error, evoking a usage statement and eventual exit
    when raised.
    '''
    def __init__(self, msg):
        super(Usage, self).__init__()
        self.msg = msg




def main():
    """Main docstring"""
    start = timer()

    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)



if __name__ == "__main__":
    main()
    raise SystemExit
