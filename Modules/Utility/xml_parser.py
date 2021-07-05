import os
import xml.etree.ElementTree as ET

import sqlite3

con = sqlite3.connect('[DATA]\db\Enzymes.db')
cur = con.cursor()

# cur.execute(
#     'CREATE TABLE Entries(EnzymeAutoID integer primary key autoincrement, accession_string str, ec_number_string str, sequence_length int, '
#     'sequence_string str )')


def process_entry(entry, ns):
    try:
        ec_number = ''
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

        cur.execute("INSERT INTO Entries(accession_string, ec_number_string, sequence_length, sequence_string) VALUES "
                    "('{0}', '{1}', '{2}', '{3}')".format(accession_string, ec_number_string, sequence_length, sequence_string))

        con.commit()

    except Exception as e:
        print(e)


def fixtag(ns, tag, nsmap):
    return '{' + nsmap[ns] + '}' + tag


# [DATA]/DummyData/ExampleDATA is dummy data of 100 entries, recorded in hte table ExampleDATA. only 90 were read succesfully.
# [DATA]/uniprot/uniprot_sprot.xml contains all the manually annotated entries wth ec number from uniprot. 271,464 entries, read  into db.

file_name = '[DATA]/uniprot/uniprot_sprot.xml'
full_file = os.path.abspath(os.path.join(file_name))

nsmap = {}

for event, elem in ET.iterparse(full_file, events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start-ns':
        ns, url = elem
        nsmap[ns] = url

    if event == 'end':
        if elem.tag == fixtag('', 'entry', nsmap):
            process_entry(elem, nsmap)
            print(elem)

            elem.clear()

print('That\'s all folks!')

cur.close()
# for row in cur.execute('SELECT * FROM Entries'):
#     print(row)
