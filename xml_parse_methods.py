'''
TODO:

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

#
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
        ec_number = recommended_name.find('{' + ns[''] + '}' + 'ecNumber')
        sequence = entry.find('{' + ns[''] + '}' + 'sequence')
        sequence_length = sequence.attrib['length']
        sequence_string = sequence.text
        accession_string = accession.text
        ec_number_string = ec_number.text


        entry_n = (accession_string, ec_number_string, sequence_length, sequence_string)
        print(entry_n)

        return entry_n


    except Exception as e:
        print(e)


def fixtag(ns, tag, nsmap):
    return '{' + nsmap[ns] + '}' + tag
