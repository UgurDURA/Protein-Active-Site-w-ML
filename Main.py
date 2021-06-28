
#This file created for trial


# To do list

'1- DATA PREPROCESSING '
' 1.1 Access the Data'
' 1.2 Spit the data into features and depandents'
' 1.3 At the end of the preprocessing the data split it into Test and Validation'
'2- TOKENIZATION'
' 2.1 BPE Algorithym'
' 2.2 '


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
# import sentencepiece as spm
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

'Data Analysis'
 
count_aminos={}
length_seqs=[]
for i, seq in enumerate(Sequence):
    length_seqs.append(len(seq))
    for a in seq:
        if a in count_aminos:
            count_aminos[a] += 1
        else:
            count_aminos[a] = 0

unique_aminos=list(count_aminos.keys())

print('Unique aminos ({}):\n{}'.format(len(unique_aminos), unique_aminos))
x=[i for i in range(len(unique_aminos))]
plt.bar(x, count_aminos.values())
plt.xticks(x, unique_aminos)
print(list(count_aminos.values())[-5:])
plt.show()


print('Average length:', np.mean(length_seqs))
print('Deviation:', np.std(length_seqs))
print('Min length:', np.min(length_seqs))
print('Max length:', np.max(length_seqs))

print('Average length:', np.mean(length_seqs))
print('Deviation:', np.std(length_seqs))
print('Min length:', np.min(length_seqs))
print('Max length:', np.max(length_seqs))

'Split dataset into test and validation'

#Important note for here, the split need to be performed after tokenization and embedding
#I just added the code here 

# from sklearn.model_selection import train_test_split

# Ec_Number_train,Ec_Number_test,Sequence_train,Sequence_test =train_test_split(Ec_Number,Sequence,test_size=0.2, random_state=1)


# print(Ec_Number_train)
# print(Ec_Number_test)
# print(Sequence_train)
# print(Sequence_test)

# def applyMask(dirtyData, dirty_idxs):
#     if (type(dirtyData)!=type(np.asarray([]))):
#         dirtyData=np.asarray(dirtyData)

#     returnData= dirtyData[np.logical_not(dirty_idxs)]

#     return returnData

# #Tokenization 

# path  = 'all.tab'
# int_path = 'interact.json'
# seq_path = 'pretraining_data.txt'
# model_path = 'm_reviewed.model'

# def filter_seqs():
# 	"""
# 	Filter sequences by length
# 	"""
	
# 	seq_ls = [seq for seq in dataset['sequence_string'] if len(seq)<1024]
# 	with open(seq_path, 'w') as filehandle:
# 		for listitem in seq_ls:
# 			filehandle.write('%s\n' % listitem)


 
# from transformer import BertForMaskedLM, BertTokenizer, pipeline
# tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
# model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
# unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
# unmasker('D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T')
 