'''
TODO:
> configure plugins
> initialize database table before insertions

> keras Transformer Model:
    - preprocess
        > divide into training/validation
    - character based tokenization
    - embedding
    - pos encoding
    - encoder & decoder blocks
        > self-attention, masking, feedforward
    - output formatting
    - train and evaluate

> parse and train in batches?

'''

import xml_parser
import os
import xml.etree.ElementTree as ET

import sqlite3
con = sqlite3.connect('Enzymes.db')
cur = con.cursor()

def process_entry(entry, ns):
    try:
        accession = entry.find('{' + ns[''] + '}' + 'accession')
        protein = entry.find('{' + ns[''] + '}' + 'protein')
        recommended_name = protein.find('{' + ns[''] + '}' + 'recommendedName')
        alternative_name = protein.findall('{' + ns[''] + '}' + 'alternativeName')
        if recommended_name.find('{' + ns[''] + '}' + 'ecNumber') is not None:
            ec_number = recommended_name.find('{' + ns[''] + '}' + 'ecNumber')
        else:
            for n in alternative_name:
                if n.find('{' + ns[''] + '}' + 'ecNumber') is not None:
                    ec_number = n.find('{' + ns[''] + '}' + 'ecNumber')
                    break
        sequence = entry.find('{' + ns[''] + '}' + 'sequence')
        sequence_length = sequence.attrib['length']
        sequence_string = sequence.text
        accession_string = accession.text
        ec_number_string = ec_number.text

        cur.execute("INSERT INTO Entries(accession_string, ec_number_string, sequence_length, sequence_string) VALUES ('{0}','{1}',{2},'{3}')".format(accession_string, ec_number_string, sequence_length, sequence_string))
        con.commit()

    except Exception as e:
        print(e)


def fixtag(ns, tag, nsmap):
    return '{' + nsmap[ns] + '}' + tag


file_name = 'ExampleDATA.xml'
full_file = os.path.abspath(os.path.join(file_name))

nsmap = {}

for event, elem in ET.iterparse(full_file, events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start-ns':
        ns, url = elem
        nsmap[ns] = url

    if event == 'end':
        if elem.tag == xml_parser.fixtag('', 'entry', nsmap):
            xml_parser.process_entry(elem, nsmap)

            elem.clear()

print('That\'s all folks!')

