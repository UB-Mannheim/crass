#!/usr/bin/env python2.7
import sys

assert sys.version_info[0]==2 and sys.version_info[1]>=7,\
    "you must install and use crass with Python version 2.7 or later, but not Python 3.x"

from distutils.core import setup

setup(
    name='crasslib',
    version='0.8',
    author='jkamlah',
    description='crass - crop and splice segments',
    packages=['crasslib'],
    entry_points={
        'console_scripts': [
            'crass = crasslib.cmd:my_console_script:crass']
    }
)
