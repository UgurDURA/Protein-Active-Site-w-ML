

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

import xml_parse_methods
import os
import xml.etree.ElementTree as ET
import sqlite3
con=sqlite3.connect('Enzymes.db')

# keras model configuration


file_name = 'uniprot-ec__+AND+reviewed_yes.xml'
full_file = os.path.abspath(os.path.join(file_name))

nsmap = {}

for event, elem in ET.iterparse(full_file, events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start-ns':
        ns, url = elem
        nsmap[ns] = url

    if event == 'end':
        if elem.tag == xml_parse_methods.fixtag('', 'entry', nsmap):
            
            xml_parse_methods.process_entry(elem, nsmap)



            
            # Sql_Query="""INSERT INTO Entries('generated_id','accession_string','ec_number_string','sequence_length','sequence_string') VALUES (%s,%s,%s,%s,%s)"""
            # cursor.execute(Sql_Query,())

            elem.clear()

print('That\'s all folks!')
