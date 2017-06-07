![crass logo](doc/img/crass_logo.png)
crop and splice segments
========================
[![Build Status](https://travis-ci.org/UB-Mannheim/crass.svg?branch=master)](https://travis-ci.org/jkamlah/UB-Mannheim/crass)
![license](https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg)

Originally written by Jan Kamlah.

Overview
--------

`crass` is a command line driven post-processing tool for scanned sheets of paper.
The main purpose is to crop segements based on separator lines and splice them afterwards
together in a certain order. In an additionally preprocessing step, `crass` might detect
the rotation of the page and will rotating it to the correct
angle. This process is called "deskewing".

Note that the automatic processing will sometimes fail. It is always a
good idea to manually control the results of `crass` and adjust the
parameter settings according to the requirements of the input. 


Building instructions
---------------------

Dependencies can be installed into a Python Virtual Environment:

    $ virtualenv crass_venv/  
    $ source crass_venv/bin/activate  
    $ pip install -r requirements.txt  
    $ python setup.py install  

An additional method using Conda is also possible:

    $ conda create -n crass_env python=2.7  
    $ source activate crass_env  
    $ conda install --file requirements.txt 
    $ python setup.py install  

Running
-------

Here is an example for a page:

    # perform deskewing, crop and splice of a page
    # start the program in the code directory (xxx/crass)
    $ cd xxx/crass
    $ python ./crass.py ./test/testimg.jpg jpg 
    
    # perform deskewing, crop and splice of a page 
    # the horziontal line is in the bottom area and is bound to the footer
    # start the program in the code directory (xxx/crass)
    $ cd xxx/crass
    $ python ./crass.py "./test/testimg_bottom_skew.jpg" "jpg" --horlinepos 2 --horlinetype 1

This will create a directory "out/..." containing all cropped
segments and debug outputs. And a subdirectory "spliced/.."
containing the final spliced image.

Further Information
-------------------

You can find more information on the [basic concepts][1] and the
[image processing][2] in the available documentation.

[1]: doc/basic-concepts.md
[2]: doc/image-processing.md
