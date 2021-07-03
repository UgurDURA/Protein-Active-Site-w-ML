import sqlite3
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd


con = sqlite3.connect('[DATA]/db/Enzymes.db')
cur = con.cursor()
 


dataset= pd.read_csv('[DATA]\DB\Entries.csv')
EcNumberDataset =list(dataset.iloc[:,2])#features
SequenceDataset =list(dataset.iloc[:,4])  #Dependent values  

print(EcNumberDataset[1])
print(SequenceDataset[1])
print(len(dataset))


EcNumberDatasetSeperated=pd.read_csv('[DATA]\DB\MainDataset\EcNumber.csv')
SequenceDatasetSpaced=pd.read_csv('[DATA]\DB\MainDataset\Sequence.csv')     

cur.execute(
    'CREATE TABLE MainDataset(AutoID integer primary key autoincrement, AccessionNumber str, EcNumberID int, SequenceID int, ECNumber int, Sequence str)')


for i in range(len(dataset)):

    cur.execute("INSERT INTO MainDataset(AccessionNumber,EcNumberID,SequenceID,ECNUmber,Sequence) VALUES "
                    "('{0}', '{1}','{2}', '{3}','{4}')".format(EcNumberDatasetSeperated.iloc[i, 1],i,i,EcNumberDatasetSeperated.iloc[i, 3],SequenceDatasetSpaced.iloc[i, 3]))
                    
con.commit()    


# 'Split dataset into test and validation'

# from sklearn.model_selection import train_test_split

# Ec_Number_train,Ec_Number_test,Sequence_train,Sequence_test =train_test_split(list(EcNumberDatasetSeperated.iloc[:,3]),list(SequenceDatasetSpaced.iloc[:,3]),test_size=0.2, random_state=1)
 
# cur.execute(
#     'CREATE TABLE TrainDataset(TrainAutoID integer primary key autoincrement,Sequence_Train str, EcNumber_Train int)')


# cur.execute(
#     'CREATE TABLE TestDataset(TestID integer primary key autoincrement,Sequence_Test str, EcNumber_Test int)')


# for i in range(len(Ec_Number_train)):

#     cur.execute("INSERT INTO TrainDataset(Sequence_Train, EcNumber_Train) VALUES "
#                     "('{0}', '{1}')".format(Sequence_train[i],Ec_Number_train[i]))
#     print(Ec_Number_train[i])


# for i in range(len(Ec_Number_test)):
#     cur.execute("INSERT INTO TestDataset(Sequence_Test,EcNumber_Test) VALUES "
#                     "('{0}', '{1}')".format(Sequence_test[i],Ec_Number_test[i]))
#     print(Ec_Number_test[i])



 

# Add Space and Seperate the EC number

# def addSpace(sequence):
#     iterable=sequence
#     separator = " " # A whitespace character.
#                 # The string to which the method will be applied
#     return separator.join(iterable)



 
# cur.execute(
#     'CREATE TABLE EcNumber(EcNumberAutoID integer primary key autoincrement,AccessionNumber str key,EcNumber_Full str, EcNumber_First int, EcNumber_Second int,EcNumber_Third int,EcNumber_Fourth int)')


# cur.execute(
#     'CREATE TABLE Sequence(SequenceAutoID integer primary key autoincrement,AccessionNumber str key, Sequence str, Sequence_Spaced str)')



# Ec_Number_FirstOnly=[]
# Ec_Number_SecondOnly=[]
# Ec_Number_ThirdOnly=[]
# Ec_Number_FourthOnly=[]
 

# for i in EcNumberDataset:

#     i=i.replace("-","-1")
#     i=i.replace("n","")
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