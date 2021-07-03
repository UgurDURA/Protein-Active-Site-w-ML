# This file is created for experimentation

# To do list
'1- DATA PREPROCESSING '
' 1.1 Access the Data'
' 1.2 Spit the data into features and depandents'
' 1.3 At the end of the preprocessing the data split it into Test and Validation'
'2- TOKENIZATION'
' 2.1 BPE Algorithym'
' 2.2 '

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

dataset = pd.read_csv('[DATA]\DB\MainDataset\MainDataset.csv')  # taking data from csv file, you can easily export the data from SQL file to csv

EcNumberDataset =list(dataset.iloc[:,4])   #features
SequenceDataset =list(dataset.iloc[:,5])   #Dependent values  

# print(EcNumberDataset)
# print(SequenceDataset)

'Data Analysis'

# count_aminos={}
# SequenceSize=len(dataset.iloc[:,-1])

# length_seqs=[]
# for i, seq in enumerate(dataset.iloc[:,-1]):
#     length_seqs.append(len(seq))
#     for a in seq:
#         if a in count_aminos:
#             count_aminos[a] += 1
#         else:
#             count_aminos[a] = 0

# unique_aminos=list(count_aminos.keys())

# print('Unique aminos ({}):\n{}'.format(len(unique_aminos), unique_aminos))
# x=[i for i in range(len(unique_aminos))]
# plt.bar(x, count_aminos.values())
# plt.xticks(x, unique_aminos)
# print(list(count_aminos.values())[-5:])
# plt.show()


# print('Average length:', np.mean(length_seqs))
# print('Deviation:', np.std(length_seqs))
# print('Min length:', np.min(length_seqs))
# print('Max length:', np.max(length_seqs))

# sorted_seqs=np.array(length_seqs)
# sorted_seqs.sort()
# print('10 shortest:\n{}\n10 longest:\n{}'.format(sorted_seqs[:10], sorted_seqs[-10:]))

# print("Number of Sequences: ", SequenceSize)
# print('Number sequences less than 30 AA:', len(sorted_seqs[sorted_seqs<30]))
# print('Number sequences more than 500 AA:', len(sorted_seqs[sorted_seqs>500]))
# print('Number sequences more than 1000 AA:', len(sorted_seqs[sorted_seqs>1000]))

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
# Optimized_length_seq=[]

# for item in length_seqs:
#     if(item<1000):
#         Optimized_length_seq.append(item)


# N_points = 10000
# n_bins = 200
# legend = ['distribution']

# fig, axs = plt.subplots(1, 1,
#     figsize =(10, 7), 
#     tight_layout = True)


# # Remove axes splines 
# for s in ['top', 'bottom', 'left', 'right']: 
#     axs.spines[s].set_visible(False) 

# # Remove x, y ticks
# axs.xaxis.set_ticks_position('none') 
# axs.yaxis.set_ticks_position('none') 

# # Add padding between axes and labels 
# axs.xaxis.set_tick_params(pad = 5) 
# axs.yaxis.set_tick_params(pad = 10) 

# # Add x, y gridlines 
# axs.grid(b = True, color ='grey', 
#         linestyle ='-.', linewidth = 0.5, 
#         alpha = 0.6) 

# # Add Text watermark 
# fig.text(0.9, 0.15, 'Proten Function Prediction', 
#          fontsize = 12, 
#          color ='red',
#          ha ='right',
#          va ='bottom', 
#          alpha = 0.7) 

# # Creating histogram
# N, bins, patches = axs.hist(Optimized_length_seq, bins = n_bins)

# # Setting color
# fracs = ((N**(1 / 5)) / N.max())
# norm = colors.Normalize(fracs.min(), fracs.max())

# for thisfrac, thispatch in zip(fracs, patches):
#     color = plt.cm.viridis(norm(thisfrac))
#     thispatch.set_facecolor(color)

# # Adding extra features    
# plt.xlabel("Sequence Length")
# plt.ylabel("Number of Sequences")
# plt.legend(legend)
# plt.title('Sequence Length Distribution')

# # Show plot
# plt.show()




# "Tokenization "
 
Sequence_Example=SequenceDataset[1]
Ec_NumberExample=EcNumberDataset[1]

print('Example Sequence:    ',Sequence_Example, " Example EC number:  ",Ec_NumberExample)


from transformers import AutoTokenizer 

MAX_LEN=512

Sequence_Example = SequenceDataset[1]
Ec_NumberExample = EcNumberDataset[1]

print('Example Sequence:    ', Sequence_Example, " Example EC number:  ", Ec_NumberExample)

tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, )
tokens=tokenizer.encode_plus(Sequence_Example, max_length=MAX_LEN,truncation=True,padding="max_length",
                                add_special_tokens=True,return_token_type_ids=False,return_attention_mask=True, return_tensors='tf')


print("TOKENS")
print(tokens)

Xids = np.zeros((len(dataset), MAX_LEN))
Xmask = np.zeros((len(dataset), MAX_LEN))

print("XIDS SHAPE")
print(Xids.shape)

for i, sequence in enumerate(dataset.iloc[:, 5]):
    tokens = tokenizer.encode_plus(sequence, max_length=MAX_LEN, truncation=True, padding="max_length",
                                   add_special_tokens=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='tf')





for i, sequence in enumerate (dataset.iloc[:,5]):
    tokens=tokenizer.encode_plus(sequence, max_length=MAX_LEN,truncation=True,padding="max_length",
                                add_special_tokens=True,return_token_type_ids=False,return_attention_mask=True, return_tensors='tf')
    
    Xids[i,:], Xmask[i,:]= tokens['input_ids'], tokens['attention_mask']


print("XIDS")
print(type(Xids))
print("XMASKS")
print(Xmask)

print(dataset.iloc[:, 4].unique)

print(dataset['ECNumber'].unique)

arr=dataset['ECNumber'].values

print("Array Size")
print(arr.size)

labels = np.zeros((arr.size, arr.max() + 1))

print("Labels Shape")
print(labels.shape)

labels[np.arange(arr.size), arr] = 1

print("LABELS")
print(labels)


# Below code is for off loading the data

with open('xids.npy','wb') as f:
    np.save(f,Xids)
with open('xmask.npy','wb') as f:
    np.save(f,Xmask)
with open('labels.npy','wb') as f:
    np.save(f,labels)



#Below code is for load the data

# with open('xids.npy','rb') as fp:
#     Xids=np.load(fp)

# with open('xmask.npy','rb') as fp:
#     Xmask=np.load(fp)

# with open('labels.npy','rb') as fp:
#     labels=np.load(fp)


tf.config.experimental.list_physical_devices('GPU')

tensorflow_dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

print("DATASET ON TENSOR FLOW EXAMPLE")
for i in tensorflow_dataset.take(1):
    print(i)


def map_func(input_ids, masks, labels):
    return{'input_ids': input_ids,'attention_mask': masks},labels


tensorflow_dataset = tensorflow_dataset.map(map_func)

for i in tensorflow_dataset.take(1):
    print(i)

tensorflow_dataset = tensorflow_dataset.shuffle(1000000).batch(32)

tensorflow_dataset=tensorflow_dataset.shuffle(100000).batch(32)

DS_LEN=len(list(tensorflow_dataset))

print(DS_LEN)

SPLIT = .9

train= tensorflow_dataset.take(round(DS_LEN*SPLIT))
val=tensorflow_dataset.skip(round(DS_LEN*SPLIT))

del tensorflow_dataset

from transformers import TFAutoModel
from tensorflow import keras

bert= TFAutoModel.from_pretrained('Rostlab/prot_bert_bfd')

input_ids=tf.keras.layers.Input(shape=(MAX_LEN,),name='input_ids', dtype='int32')
mask=tf.keras.layers.Input(shape=(MAX_LEN,),name='attention_mask', dtype='int32')

input_ids = tf.keras.layers.Input(shape=(MAX_LEN), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(MAX_LEN), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]

X=tf.keras.layers.GlobalMaxPooling1D()(embeddings)
X=tf.keras.layers.BatchNormalization()(X)
X=tf.keras.layers.Dense(128,activation='relu')(X)
X=tf.keras.layers.Dropout(0.1)(X)
X=tf.keras.layers.Dense(32,activation='relu')(X)
y=tf.keras.layers.Dense(8,activation='softmax', name='outputs')(X)


model= tf.keras.Model(inputs=[input_ids, mask], outputs=[y])

model.layers[2].trainable=False
model.summary()

optimizer= tf.keras.optimizers.Adam(0.01)
loss= tf.keras.losses.CategoricalCrossentropy()
acc= tf.keras.metrics.CategoricalAccuracy('accuracy')


model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

history = model.fit(

    train,
    validation_data=val,
    epochs=100,

)

print(history)
