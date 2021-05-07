'''
TODO:
> configure plugins
> initialize database table before insertions

> keras Transformer Model:
    - preprocess
        > divide into training/validation
    - embedding
    - pos embedding
    - encoder & decoder
        > self-attention, masking, feedforward
    - output formatting
    - train and evaluate

> parse and train in batches?

'''

#import os
import sqlite3
con = sqlite3.connect('Enzymes.db')
cur = con.cursor()

# file_name = 'uniprot-ec__+AND+reviewed_yes.xml'
# full_file = os.path.abspath(os.path.join(file_name))


# def get_string_or_none(element_list, delimiter):
#     list_text = ''
#     try:
#         for element in element_list:
#             list_text += delimiter + element.text
#         list_text = list_text.replace(delimiter, '', 1)
#         if delimiter in list_text:
#             list_text = '"' + list_text + '"'
#     except AttributeError:
#         print('attrib err')
#     return list_text
#
#
# def list_to_str(string_list, delimiter):
#     joined_string = delimiter.join(string_list)
#     if delimiter in joined_string:
#         joined_string = '"' + joined_string + '"'
#     return joined_string


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
