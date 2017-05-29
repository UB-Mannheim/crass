#!/usr/bin/env python
import sys

assert sys.version_info[0]==2 and sys.version_info[1]>=7,\
    "you must install and use crass with Python version 2.7 or later, but not Python 3.x"

from distutils.core import setup

setup(
    name='crass',
    version='0.8',
    packages=[''],
    author='jkamlah',
    description='crass - crop and splice segments'
)
