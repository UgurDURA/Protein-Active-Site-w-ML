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
max_seq = 0
for event, elem in ET.iterparse(full_file, events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start-ns':
        ns, url = elem
        nsmap[ns] = url

    if event == 'end':
        if elem.tag == fixtag('', 'entry', nsmap):
            e = int(process_entry(elem, nsmap))
            if e > max_seq:
                max_seq = e
                print(max_seq)
            elem.clear()

print(max_seq)
print('That\'s all folks!')

# 1206
# 2337
# 2382
# 3100
# 18562
# 35213
# 35213
# That's all folks!