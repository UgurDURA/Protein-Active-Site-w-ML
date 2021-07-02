import sqlite3
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd


con = sqlite3.connect('[DATA]/db/Enzymes.db')
cur = con.cursor()
 



EcNumberDatasetSeperated=pd.read_csv('[DATA]\DB\EcNumber\EcNumber.csv')
SequenceDatasetSpaced=pd.read_csv('[DATA]\DB\Sequence\Sequence.csv')     #a better name could be suggested


'Split dataset into test and validation'

#Important note for here, the split need to be performed after tokenization and embedding
#I just added the code here 

from sklearn.model_selection import train_test_split

Ec_Number_train,Ec_Number_test,Sequence_train,Sequence_test =train_test_split(EcNumberDatasetSeperated.iloc[:,3],SequenceDatasetSpaced.iloc[:,3],test_size=0.2, random_state=1)
 
cur.execute(
    'CREATE TABLE TrainDataset(TrainAutoID integer primary key autoincrement,Sequence_Train str, EcNumber_Train int)')
print(Ec_Number_train)


cur.execute("INSERT INTO TrainDataset(Sequence_Train, EcNumber_Train) VALUES "
                    "('{0}', '{1}')".format(Sequence_train,Ec_Number_train))


cur.execute(
    'CREATE TABLE TestDataset(TestID integer primary key autoincrement,Sequence_Test str, EcNumber_Test int)')

print(Ec_Number_test)
cur.execute("INSERT INTO TestDataset(Sequence_Test,EcNumber_Test) VALUES "
                    "('{0}', '{1}')".format(Sequence_test,Ec_Number_test))

con.commit()








# def addSpace(sequence):
#     iterable=sequence
#     separator = " " # A whitespace character.
#                 # The string to which the method will be applied
#     return separator.join(iterable)



 
# cur.execute(
#     'CREATE TABLE EcNumber(EcNumberAutoID integer primary key autoincrement,AccessionNumber str key,EcNumber_Full str, EcNumber_First int, EcNumber_Second int,EcNumber_Third int,EcNumber_Fourth int)')


# cur.execute(
#     'CREATE TABLE Sequence(EcNumberAutoID integer primary key autoincrement,AccessionNumber str key, Sequence str, Sequence_Spaced str)')



# Ec_Number_FirstOnly=[]
# Ec_Number_SecondOnly=[]
# Ec_Number_ThirdOnly=[]
# Ec_Number_FourthOnly=[]
 

# for i in EcNumberDataset:

#     i=i.replace("-","-1")
#     print(i)

#     Seperated_EcNumber= i.split('.')

#     Ec_Number_FirstOnly.append(int(Seperated_EcNumber[0]))
#     Ec_Number_SecondOnly.append(int(Seperated_EcNumber[1]))
#     Ec_Number_ThirdOnly.append(int(Seperated_EcNumber[2]))
#     Ec_Number_FourthOnly.append(int(Seperated_EcNumber[3]))

# for i in range(len(dataset)):
#     EcNumberDataset[i]=EcNumberDataset[i].replace("-","-1")
#     cur.execute("INSERT INTO EcNumber(AccessionNumber,EcNumber_Full, EcNumber_First, EcNumber_Second, EcNumber_Third, EcNumber_Fourth) VALUES "
#                     "('{0}','{1}','{2}','{3}','{4}','{5}')".format(dataset.iloc[i,1],EcNumberDataset[i],Ec_Number_FirstOnly[i],Ec_Number_SecondOnly[i],Ec_Number_ThirdOnly[i],Ec_Number_FourthOnly[i]))
#     con.commit()

#     cur.execute("INSERT INTO Sequence(AccessionNumber, Sequence, Sequence_Spaced) VALUES "
#                     "('{0}','{1}','{2}')".format(dataset.iloc[i,1],SequenceDataset[i],addSpace(SequenceDataset[i])))
#     con.commit()





# print(Ec_Number_FirstOnly)
# print(Ec_Number_SecondOnly)
# print(Ec_Number_ThirdOnly)
# print(Ec_Number_FourthOnly)