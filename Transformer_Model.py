

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

import xml_parse_methods
import os
import xml.etree.ElementTree as ET

# keras model configuration


file_name = 'uniprot-ec__+AND+reviewed_yes.xml'
full_file = os.path.abspath(os.path.join(file_name))

nsmap = {}
dataset = []

for event, elem in ET.iterparse(full_file, events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start-ns':
        ns, url = elem
        nsmap[ns] = url

    if event == 'end':
        if elem.tag == xml_parse_methods.fixtag('', 'entry', nsmap):
            dataset += xml_parse_methods.process_entry(elem, nsmap)
            elem.clear()

print(dataset)
print('That\'s all folks!')
