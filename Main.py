
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
X= dataset.values #a better name could be suggested

print(X)














#Split dataset into test and validation

#Important note for here, the split need to be performed after tokenization and embedding
#I just added the code here 

# from sklearn.model_selection import train_test_split

# X_train,X_test =train_test_split(X,test_size=0.2, random_state=1)

 