'''
scriptlet to divide and preprocess the entries in our database.
as preprocessing, spaces are added inbetween every token in a sequence : addSpaces();
additionally, EC numbers are divided into the digits by seperator '.' : ECnumberSeperator
'''

from ... import *

# simple functions to add space between AA sequences, and seperating the EC numbers:

def addSpaces(sequence):
    iterable = sequence    # The string to which the method will be applied
    separator = " "  # A whitespace character.
    return separator.join(iterable)


def ECnumberSeperator(ECnumber):
    '''
    for now, seperates the numbers and returns them.
    ennumerates N/A numbers into '0', will serve as only first number classification's output encoding.
    '''

    ECnumber = ECnumber.replace("n", "")
    ECnumber = ECnumber.replace("-", "0")
    print('EC Number: ' + ECnumber + '\n')

    seperatedECnumber = ECnumber.split('.')

    return seperatedECnumber[0], seperatedECnumber[1], seperatedECnumber[2], seperatedECnumber[3]

def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

def main():

    con = sqlite3.connect(SQLite_DB_PATH)
    cur1 = con.cursor()
    cur2 = con.cursor()

    # dataset = pd.read_csv('../../[DATA]/DummyData/ExampleDATA.csv')
    # EcNumberDataset = list(dataset.iloc[:, 2])  # features         : ec_number_string
    # SequenceDataset = list(dataset.iloc[:, 4])  # Dependent values : sequence_string
    #
    # print(EcNumberDataset[1])
    # print(SequenceDataset[1])
    # print(len(dataset))

    cur1.execute("SELECT EnzymeAutoID, accession_string, ec_number_string, sequence_string FROM Entries;")
    rows = cur1.fetchall()
    print('length of table: ' + str(len(rows)))

    for i in rows:
        ec_1, ec_2, ec_3, ec_4 = ECnumberSeperator(i[2])

        cur2.execute("INSERT OR REPLACE INTO EntriesReady(EnzymeAutoID, accession_string, ec_number_one, ec_number_two, ec_number_three, "
                     "ec_number_four, sequence_string) VALUES ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}')".format(i[0], i[1], ec_1, ec_2, ec_3,
                                                                                                                        ec_4, addSpaces(i[3])))
        con.commit()

    cur1.close()
    cur2.close()
    con.close()


if __name__ == "__main__":
    main()
