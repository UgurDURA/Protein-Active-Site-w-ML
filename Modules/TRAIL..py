from enum import unique
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import sqlite3
import os
import transformers
import matplotlib.pyplot as plt

MAX_LEN = 512
BATCH_SIZE = 16  # Possible Values: 4/8/16/32
DATA_SIZE = 100

con = sqlite3.connect(r'[DATA]\Enzymes.db')

dataset = pd.read_sql_query("SELECT ec_number_one, ec_number_two, sequence_string FROM EntriesReady LIMIT ('{0}')".format(DATA_SIZE), con)

print(dataset)

tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, )

Xids = np.zeros((len(dataset), MAX_LEN))
Xmask = np.zeros((len(dataset), MAX_LEN))

print("XIDS SHAPE")
print(Xids.shape)

for i, sequence in enumerate(dataset['sequence_string']):
    tokens = tokenizer.encode_plus(sequence, max_length=MAX_LEN, truncation=True, padding="max_length",
                                   add_special_tokens=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='tf')

    Xids[i, :], Xmask[i, :] = tokens['input_ids'], tokens['attention_mask']

print("XIDS")
print(type(Xids))
print("XMASKS")
print(Xmask)


Accumulated_EC=[]
First_EC_List= list(dataset['ec_number_one'])
Second_EC_List=list(dataset['ec_number_two'])

from sklearn import svm

plt.scatter(First_EC_List,Second_EC_List)
plt.show()

print(First_EC_List)
print(Second_EC_List)

for i in range (len(dataset['ec_number_one'])):
    Accumulated_EC.append(int(str(First_EC_List[i])+"666"+ str(Second_EC_List[i])))


 
# print(Accumulated_EC)

# arr =np.array(Accumulated_EC)
# UniqueArr=np.unique(arr)
# print("Unique ARR", UniqueArr)
# print("Leng of the Unique arr: ", len(UniqueArr))
# print("Array Size")
# print(arr.size)
# print(UniqueArr)

# labels = np.zeros((arr.size, (UniqueArr.size)+1))

# print("Labels Shape")
# print(labels.shape)

# labels[np.arange(arr.size), arr] = 1

# print("LABELS")
# print(labels)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
arr= np.array(ct.fit_transform(Accumulated_EC))


labels=arr



print("Labels Shape")
print(labels.shape)

 

print("LABELS")
print(labels)

# # Below code is for off loading the data

# with open('xids.npy','wb') as f:
#     np.save(f,Xids)
# with open('xmask.npy','wb') as f:
#     np.save(f,Xmask)
# with open('labels.npy','wb') as f:
#     np.save(f,labels)


# Below code is for load the data

# with open(r'C:\Users\ugur_\Desktop\Projects\Protein-Active-Site-w-ML\xids.npy','rb') as fp:
#     Xids=np.load(fp)

# with open(r'C:\Users\ugur_\Desktop\Projects\Protein-Active-Site-w-ML\xmask.npy','rb') as fp:
#     Xmask=np.load(fp)

# with open(r'C:\Users\ugur_\Desktop\Projects\Protein-Active-Site-w-ML\labels.npy','rb') as fp:
#     labels=np.load(fp)

tensorflow_dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

print("DATASET ON TENSOR FLOW EXAMPLE")
for i in tensorflow_dataset.take(1):
    print(i)


def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


tensorflow_dataset = tensorflow_dataset.map(map_func)

for i in tensorflow_dataset.take(1):
    print(i)

tensorflow_dataset = tensorflow_dataset.shuffle(100000).batch(4)

DS_LEN = len(list(tensorflow_dataset))

print(DS_LEN)

SPLIT = .9

train = tensorflow_dataset.take(round(DS_LEN * SPLIT))
val = tensorflow_dataset.skip(round(DS_LEN * SPLIT))


bert = TFAutoModel.from_pretrained('Rostlab/prot_bert_bfd')

input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]

X = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation='relu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(32, activation='relu')(X)
y = tf.keras.layers.Dense((UniqueArr.size)+1, activation='softmax', name='outputs')(X)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=[y])

model.layers[2].trainable = False
model.summary()

optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

history = model.fit(
    train,
    validation_data=val,
    epochs=15,
)
