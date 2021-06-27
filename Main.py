
#This file created for trial

import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from tensorflow.keras import layers

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


dataset= pd.read_csv('[DATA]/DB/Entries.csv')  #taking data from csv file, you can easily export the data from SQL file to csv
Ec_Number = dataset.iloc[:,2].values #features
Sequence = dataset.iloc[:,-1].values  #Dependent values      #a better name could be suggested

print(Ec_Number)
print(Sequence)

#Split dataset into test and validation

#Important note for here, the split need to be performed after tokenization and embedding
#I just added the code here 

from sklearn.model_selection import train_test_split

Ec_Number_train,Ec_Number_test,Sequence_train,Sequence_test =train_test_split(Ec_Number,Sequence,test_size=0.2, random_state=1)


print(Ec_Number_train)
print(Ec_Number_test)
print(Sequence_train)
print(Sequence_test)

for _examples, en_examples in train_examples.batch(3).take(1):
  for pt in pt_examples.numpy():
    print(pt.decode('utf-8'))

  print()

  for en in en_examples.numpy():
    print(en.decode('utf-8'))