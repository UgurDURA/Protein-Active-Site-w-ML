

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

import xml_parse_methods
import os
import xml.etree.ElementTree as ET

# keras model configuration


file_name = 'ExampleDATA.xml'
full_file = os.path.abspath(os.path.join(file_name))

nsmap = {}

for event, elem in ET.iterparse(full_file, events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start-ns':
        ns, url = elem
        nsmap[ns] = url

    if event == 'end':
        if elem.tag == xml_parse_methods.fixtag('', 'entry', nsmap):
            
            xml_parse_methods.process_entry(elem, nsmap)

            elem.clear()

print('That\'s all folks!')
