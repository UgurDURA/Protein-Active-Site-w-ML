
#imports to be used - subject to change
"""
import os
import numpy as NP
import pandas as pds
import matplotlib as MP

import sklearn as skl
import tensorflow as TF
import torch
import logging
"""

import os
from xml.etree import ElementTree as ET

file_name = 'uniprot.xml'
full_file = os.path.abspath(os.path.join(file_name))
print("file opened.")

#root = ET.parse(full_file).getroot()
#parser = ET.XMLPullParser(['start', 'end'] )     # nonblocking read

input_seq = []
output_ec = []
print("lists initialized.")

print("starting iteration.")
iterator = ET.iterparse(full_file, events=('start', 'end'))      # blocking read
for event, elem in iterator:
    print(elem.tag, elem.text, elem.attrib, '\n')
    if elem.attrib == 'sequence':
        input_seq = input_seq.append(elem.attrib.text)


    '''
    if elem.tag == 'entry':
        # print("event: ", event)
        print(elem.attrib)
'''

# two main attributes.
# input, aka the aa sequence.
# output, aka the EC number.
# + Accession Number, for identification of entries.




"""
https://docs.python.org/3/library/xml.etree.elementtree.html#module-xml.etree.ElementTree

feed(data)
close()
read_events()
"""
