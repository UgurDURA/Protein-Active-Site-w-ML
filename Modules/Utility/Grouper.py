"""
scriptlet to save the autoID's of the entries into text files;
named second.txt, third.txt and fourth.txt (stored in "EC Groupings" folder), if they have the first
two, three or all numbers in EC number notation respectively,

when going to use, read EnzymeAutoID's from the text file. each line is a new entry.
after reading all into a list, divide into training/validation or just access entries from the database via EnzymeAutoID.
"""

import sqlite3
import re
SQLite_DB_PATH = r'[DATA]/Enzymes.db'


def main():
    con = sqlite3.connect(SQLite_DB_PATH)
    cur1 = con.cursor()

    # cur1.execute("SELECT * FROM Entries")

    cur1.execute("SELECT EnzymeAutoID FROM EntriesReady WHERE ec_number_one=1;")

    ID_list =list(cur1.fetchall())

    print(len(ID_list))

    # while True:
    #     current_row = cur1.fetchone()
    #
    #     if not current_row:
    #         break
    #
    #     ec_num = current_row[2]
    #     autoID = current_row[0]
    #     # acc_str = current_row[1]
    #     # seq_len = current_row[3]
    #     # seq_str = current_row[4]
    #
    #     '''
    #     perform regex matching:
    #     first two = '\d+\.\d+\.*'
    #     first three = '\d+\.\d+\.\d+\.*'
    #     all four = '\d+\.\d+\.\d+\.\d+'
    #
    #     '''
    #
    #     match_str = '\d+\.\d+\.\d+\.*'
    #
    #     if re.match(match_str, ec_num):
    #         print('matched!')
    #         print(ec_num)
    #
    #         ID_list.append(autoID)
    #
    #         # # insert into appropriate table
    #         # cur2 = con.cursor()
    #         # cur2.execute("INSERT OR REPLACE INTO Fourth(EnzymeAutoID, accession_string, ec_number_string, sequence_length, sequence_string) "
    #         #              "VALUES ('{0}', '{1}', '{2}', '{3}', '{4}')".format(autoID, acc_str, ec_num, seq_len, seq_str))
    #         # con.commit()
    #         # cur2.close()

    # save autoID's to appropriate file
    out_file = open(r"EC Groupings/1.txt", "w+")

    for ID in ID_list:
        out_file.write(str(ID) + "\n")

    # each EnzymeAutoID in a new line. read them line by line from file when going to use in training.
    # important: for each group, let script run once unimpeded. this script does not check for duplicates and the output will be in ascending order.

    cur1.close()
    con.close()
    out_file.close()


if __name__ == "__main__":
    main()

"""
first digit == 1 : 32701
first digit == 2 : 92322
first digit == 3 : 57912
first digit == 4 : 24281
first digit == 5 : 14019
first digit == 6 : 28655
first digit == 7 : 14630
"""
