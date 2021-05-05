'''
TODO:
> edit process entry to add relevant attributes to a container

> keras Transformer Model:
    - preprocess
    - embedding
    - pos embedding
    - encoder & decoder
     - self-attention, masking,feedforward
    - output formatting

'''


import os
import xml.etree.ElementTree as etree

file_name = 'ExampleDATA.xml'
full_file = os.path.abspath(os.path.join(file_name))


def get_string_or_none(element_list, delimiter):
    list_text = ''
    try:
        for element in element_list:
            list_text += delimiter + element.text
        list_text = list_text.replace(delimiter, '', 1)
        if delimiter in list_text:
            list_text = '"' + list_text + '"'
    except AttributeError:
        print('attrib err')
    return list_text


def list_to_str(string_list, delimiter):
    joined_string = delimiter.join(string_list)
    if delimiter in joined_string:
        joined_string = '"' + joined_string + '"'
    return joined_string


def process_entry(entry, ns):
    try:
        accession = entry.find('{' + ns[''] + '}' + 'accession')
        protein = entry.find('{' + ns[''] + '}' + 'protein')
        recommended_name = protein.find('{' + ns[''] + '}' + 'recommendedName')
        ec_numbers = recommended_name.findall('{' + ns[''] + '}' + 'ecNumber')
        sequence = entry.find('{' + ns[''] + '}' + 'sequence')
        sequence_length = sequence.attrib['length']
        sequence_string = sequence.text
        accesion_string = accession.text
        ec_numbers_string = get_string_or_none(ec_numbers, '#')

        print(accesion_string + ', ' + ec_numbers_string + ', ' + sequence_length)
        print(sequence_string)
    except Exception as e:
        print(e)


def fixtag(ns, tag, nsmap):
    return '{' + nsmap[ns] + '}' + tag
