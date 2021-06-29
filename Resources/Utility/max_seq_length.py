import os
import xml.etree.ElementTree as ET


def process_entry(entry, ns):
    try:
        sequence = entry.find('{' + ns[''] + '}' + 'sequence')
        sequence_length = sequence.attrib['length']

        return sequence_length

    except Exception as e:
        print(e)


def fixtag(ns, tag, nsmap):
    return '{' + nsmap[ns] + '}' + tag


file_name = 'uniprot.xml'
full_file = os.path.abspath(os.path.join(file_name))

nsmap = {}
seq_counter = 0
for event, elem in ET.iterparse(full_file, events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start-ns':
        ns, url = elem
        nsmap[ns] = url

    if event == 'end':
        if elem.tag == fixtag('', 'entry', nsmap):
            e = int(process_entry(elem, nsmap))
            if e > 3000:
                seq_counter = seq_counter + 1
                print(seq_counter)
            elem.clear()

print(seq_counter)
print('That\'s all folks!')


# max sequence length: 35213
# 451 entries with sequence length longer than 3000.