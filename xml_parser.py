
import os
import xml.etree.ElementTree as ET

import sqlite3

con = sqlite3.connect('[DATA]\DB\EnzymeData.db')
cur = con.cursor()
# cur.execute('''CREATE TABLE Enzymes
#              (EnzymeAutoID int IDENTITY(1,1) PRIMARY KEY,[accession_string] text, [ec_number_string] text,[sequence_length] integer,[sequence_string] text )''')

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


file_name = '[DATA]\UniProt\uniprot_sprot.xml'
full_file = os.path.abspath(os.path.join(file_name))

nsmap = {}

for event, elem in ET.iterparse(full_file, events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'start-ns':
        ns, url = elem
        nsmap[ns] = url

    if event == 'end':
        if elem.tag == fixtag('', 'entry', nsmap):
            process_entry(elem, nsmap)

            elem.clear()

print('That\'s all folks!')
for row in cur.execute('SELECT rowid, * FROM Entries'):
    print(row)
