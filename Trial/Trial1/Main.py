
#This file is created for experimentation

# To do list
'1- DATA PREPROCESSING '
' 1.1 Access the Data'
' 1.2 Spit the data into features and depandents'
' 1.3 At the end of the preprocessing the data split it into Test and Validation'
'2- TOKENIZATION'
' 2.1 BPE Algorithym'
' 2.2 '

from ast import increment_lineno
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
from matplotlib import colors

import numpy as np
import pandas as pd
import scipy as sp
# import sentencepiece as spm
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
import sqlite3
from tensorflow.keras import layers

 
 
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


dataset= pd.read_csv('[DATA]\DummyData\TestData.csv')  #taking data from csv file, you can easily export the data from SQL file to csv
EcNumberDataset = list(dataset.iloc[:,2].values)#features
SequenceDataset = list(dataset.iloc[:,-1].values)  #Dependent values      #a better name could be suggested

EcNumberDatasetSeperated=pd.read_csv('[DATA]\DB\EcNumber\EcNumber.csv')
SequenceDatasetSpaced=pd.read_csv('[DATA]\DB\Sequence\Sequence.csv')

'Data Analysis'

count_aminos={}
SequenceSize=len(SequenceDataset)

length_seqs=[]
for i, seq in enumerate(SequenceDataset):
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

sorted_seqs=np.array(length_seqs)
sorted_seqs.sort()
print('10 shortest:\n{}\n10 longest:\n{}'.format(sorted_seqs[:10], sorted_seqs[-10:]))

print("Number of Sequences: ", SequenceSize)
print('Number sequences less than 30 AA:', len(sorted_seqs[sorted_seqs<30]))
print('Number sequences more than 500 AA:', len(sorted_seqs[sorted_seqs>500]))
print('Number sequences more than 1000 AA:', len(sorted_seqs[sorted_seqs>1000]))

# density={}

# for i in range(np.max(length_seqs)):
#     lower=len(sorted_seqs[sorted_seqs<i])
#     upper=len(sorted_seqs[sorted_seqs>i+2])

#     print ("Lower  ",lower,"    ","Upper   ",upper,"       ","Sequence Length",np.max(length_seqs))
    
#     calc=(SequenceSize)-abs(lower)-abs(upper)
#     calc=abs(calc)
    
#     density[i]=calc
#     # print(i, " / ", np.max(length_seqs))

# lists = sorted(density.items()) # sorted by key, return a list of tuples
# print(density)
# x, y = zip(*lists) # unpack a list of pairs into two tuples



# plt.plot(x, y)
# plt.xlabel('Number of Sequencees')
# plt.ylabel('The Length of the Sequence')
# plt.title('Sequence Size Distribution')

# plt.show()
Optimized_length_seq=[]

for item in length_seqs:
    if(item<1000):
        Optimized_length_seq.append(item)
    

N_points = 10000
n_bins = 200
legend = ['distribution']

fig, axs = plt.subplots(1, 1,
    figsize =(10, 7), 
    tight_layout = True)
  
  
# Remove axes splines 
for s in ['top', 'bottom', 'left', 'right']: 
    axs.spines[s].set_visible(False) 
  
# Remove x, y ticks
axs.xaxis.set_ticks_position('none') 
axs.yaxis.set_ticks_position('none') 
    
# Add padding between axes and labels 
axs.xaxis.set_tick_params(pad = 5) 
axs.yaxis.set_tick_params(pad = 10) 
  
# Add x, y gridlines 
axs.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.5, 
        alpha = 0.6) 
  
# Add Text watermark 
fig.text(0.9, 0.15, 'Proten Function Prediction', 
         fontsize = 12, 
         color ='red',
         ha ='right',
         va ='bottom', 
         alpha = 0.7) 
  
# Creating histogram
N, bins, patches = axs.hist(Optimized_length_seq, bins = n_bins)
  
# Setting color
fracs = ((N**(1 / 5)) / N.max())
norm = colors.Normalize(fracs.min(), fracs.max())
  
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
  
# Adding extra features    
plt.xlabel("Sequence Length")
plt.ylabel("Number of Sequences")
plt.legend(legend)
plt.title('Sequence Length Distribution')
  
# Show plot
plt.show()

 
 
    

'Split dataset into test and validation'

#Important note for here, the split need to be performed after tokenization and embedding
#I just added the code here 

from sklearn.model_selection import train_test_split

Ec_Number_train,Ec_Number_test,Sequence_train,Sequence_test =train_test_split(EcNumberDatasetSeperated.iloc[:,3],SequenceDatasetSpaced.iloc[:,3],test_size=0.2, random_state=1)




"Tokenization "
def addSpace(sequence):
    iterable=sequence
    separator = " " # A whitespace character.
                # The string to which the method will be applied
    return separator.join(iterable)


    
Sequence_Example=SequenceDataset[1]
Ec_NumberExample=EcNumberDataset[1]

print('Example Sequence:    ',addSpace(Sequence_Example), " Example EC number:  ",EC_First(Ec_NumberExample))


from transformers import BertTokenizer 

MAX_LEN=512




tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd_localization', do_lower_case=False, )
tokens=tokenizer.encode_plus(addSpace(Sequence_Example), max_length=MAX_LEN,truncation=True,padding="max_length",
                                add_special_tokens=True,return_token_type_ids=False,return_attention_mask=True, return_tensors='tf')

print(tokens)


Xids= np.zeros((len(dataset),MAX_LEN))
Xmask= np.zeros((len(dataset),MAX_LEN))

print(Xids.shape)





for i, sequence in enumerate (dataset['sequence_string']):
    tokens=tokenizer.encode_plus(addSpace(sequence), max_length=MAX_LEN,truncation=True,padding="max_length",
                                add_special_tokens=True,return_token_type_ids=False,return_attention_mask=True, return_tensors='tf')
    
    Xids[i,:], Xmask[i,:]= tokens['input_ids'], tokens['attention_mask']

print(Xids)
print(Xmask)


print(Ec_Number_FirstOnly)

arr=Ec_Number_FirstOnly

labels=np.zeros((arr.size,arr.max()+1))

print(labels.shape)





# tokens = tokenizers.en.lookup(encoded)
# print(tokens)


# SequenceDataset=np.array(SequenceDataset)

# Result=[]

# for SequenceTokens in SequenceDataset:



#     sequence_Example = re.sub(r"[UZOB]", "X", SequenceTokens)
#     encoded_input = tokenizer(sequence_Example, return_tensors='pt')
#     output = model_name(**encoded_input)
#     print(output)
    


# print(Result)


#Setup input pipeline











