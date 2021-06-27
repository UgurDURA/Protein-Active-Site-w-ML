
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
import scipy as sp
import sentencepiece as spm
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

def applyMask(dirtyData, dirty_idxs):
    if (type(dirtyData)!=type(np.asarray([]))):
        dirtyData=np.asarray(dirtyData)

    returnData= dirtyData[np.logical_not(dirty_idxs)]

    return returnData

#Tokenization 

path  = 'all.tab'
int_path = 'interact.json'
seq_path = 'pretraining_data.txt'
model_path = 'm_reviewed.model'

def filter_seqs():
	"""
	Filter sequences by length
	"""
	
	seq_ls = [seq for seq in dataset['sequence_string'] if len(seq)<1024]
	with open(seq_path, 'w') as filehandle:
		for listitem in seq_ls:
			filehandle.write('%s\n' % listitem)


 
from transformer import BertForMaskedLM, BertTokenizer, pipeline
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
unmasker('D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T')
 