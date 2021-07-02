import sqlite3
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd


con = sqlite3.connect('[DATA]/db/Enzymes.db')
cur = con.cursor()
 



dataset= pd.read_csv('[DATA]\DummyData\TestData.csv')  #taking data from csv file, you can easily export the data from SQL file to csv
EcNumberDataset = list(dataset.iloc[:,2].values)#features
SequenceDataset = list(dataset.iloc[:,-1].values)  #Dependent values      #a better name could be suggested

 
# cur.execute(
#     'CREATE TABLE EcNumber(EcNumberAutoID integer primary key autoincrement,EcNumber_Full str, EcNumber_First int, EcNumber_Second int,EcNumber_Third int,EcNumber_Fourth int)')



Ec_Number_FirstOnly=[]
Ec_Number_SecondOnly=[]
Ec_Number_ThirdOnly=[]
Ec_Number_FourthOnly=[]
 

for i in EcNumberDataset:

    i=i.replace("-","-1")
    print(i)

    Seperated_EcNumber= i.split('.')

    Ec_Number_FirstOnly.append(int(Seperated_EcNumber[0]))
    Ec_Number_SecondOnly.append(int(Seperated_EcNumber[1]))
    Ec_Number_ThirdOnly.append(int(Seperated_EcNumber[2]))
    Ec_Number_FourthOnly.append(int(Seperated_EcNumber[3]))

# for i in range(len(EcNumberDataset)):
#     EcNumberDataset[i]=EcNumberDataset[i].replace("-","-1")
#     cur.execute("INSERT INTO EcNumber(EcNumber_Full, EcNumber_First, EcNumber_Second, EcNumber_Third, EcNumber_Fourth) VALUES "
#                     "('{0}','{1}','{2}','{3}','{4}')".format(EcNumberDataset[i],Ec_Number_FirstOnly[i],Ec_Number_SecondOnly[i],Ec_Number_ThirdOnly[i],Ec_Number_FourthOnly[i]))
#     con.commit()





print(Ec_Number_FirstOnly)
print(Ec_Number_SecondOnly)
print(Ec_Number_ThirdOnly)
print(Ec_Number_FourthOnly)