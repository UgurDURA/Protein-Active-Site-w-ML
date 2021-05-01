'''
TODO:
> edit process entry to add relevant attributes to a container
>

> pyTorch Transformer Model
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
        name = entry.find('{' + ns[''] + '}' + 'name')
        protein = entry.find('{' + ns[''] + '}' + 'protein')
        recommended_name = protein.find('{' + ns[''] + '}' + 'recommendedName')
        full_name = recommended_name.find('{' + ns[''] + '}' + 'fullName')
        ec_numbers = recommended_name.findall('{' + ns[''] + '}' + 'ecNumber')
        # rhea_ids = []
        # for comment in entry.findall('{' + ns[''] + '}' + 'comment'):
        #     try:
        #         if comment.attrib['type'] == 'catalytic activity':
        #             reactions = comment.findall('{' + ns[''] + '}' + 'reaction')
        #             for reaction in reactions:
        #                 for db_reference in reaction.findall('{' + ns[''] + '}' + 'dbReference'):
        #                     if db_reference.attrib['type'] == 'Rhea':
        #                         rhea_ids.append(db_reference.attrib['id'])
        #     except AttributeError as e:
        #         print('Exception in reactions' + str(e))
        # features = entry.findall('{' + ns[''] + '}' + 'feature')
        # feature_list = []
        # for feature in features:
        #     location = feature.find('{' + ns[''] + '}' + 'location')
        #     position = location.find('{' + ns[''] + '}' + 'position')
        #     if position is not None and len(feature.attrib['description']) < 40:
        #         feature_str = feature.attrib['type'] + ',' + feature.attrib['description'] + ',' + position.attrib[
        #             'position']
        #         feature_list.append(feature_str)
        # features_string = list_to_str(feature_list, '#')
        sequence = entry.find('{' + ns[''] + '}' + 'sequence')
        sequence_length = sequence.attrib['length']
        sequence_string = sequence.text
        accesion_string = accession.text
        full_name_string = full_name.text
        ec_numbers_string = get_string_or_none(ec_numbers, '#')
        #rhea_ids_string = list_to_str(rhea_ids, '#')
        #    print(
        #        accesion_string + ', ' + full_name_string + ', ' + ec_numbers_string + ', ' + rhea_ids_string + ', ' + features_string + ', ' + sequence_length + ', ' + sequence_string)

        print(accesion_string + ', ' + ec_numbers_string + ', ' + sequence_length)
        print(sequence_string)
    except Exception as e:
        print(e)


def fixtag(ns, tag, nsmap):
    return '{' + nsmap[ns] + '}' + tag

'''
nsmap = {}
for event, elem in etree.iterparse(full_file, events=('start-ns',)):
    ns, url = elem
    nsmap[ns] = url
    print('a')

for event, elem in etree.iterparse(full_file, events=('end', 'start')):
    if event == 'end':
        if elem.tag == fixtag('', 'entry', nsmap):
            process_entry(elem, nsmap)
            elem.clear()
            print('b')
'''