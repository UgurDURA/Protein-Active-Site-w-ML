import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import sqlite3
import matplotlib.pyplot as plt
import re

MAX_LEN = 512
BATCH_SIZE = 64 # Possible Values: 4/8/16/32
DATA_SIZE =60000
con = sqlite3.connect('[DATA]\Enzymes.db')

dataset = pd.read_sql_query("SELECT ec_number_one, sequence_string FROM EntriesReady LIMIT ('{0}')".format(DATA_SIZE), con)


tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, )

Xids = np.zeros((len(dataset), MAX_LEN))
Xmask = np.zeros((len(dataset), MAX_LEN))

#print("XIDS SHAPE")
#print(Xids.shape)

for i, sequence in enumerate(dataset['sequence_string']):

    sequence=re.sub(r"[UZOB]", "X", sequence)
    tokens = tokenizer.encode_plus(sequence, max_length=MAX_LEN, truncation=True, padding="max_length",
                                   add_special_tokens=False, return_token_type_ids=False, return_attention_mask=True, return_tensors='tf')

    Xids[i, :], Xmask[i, :] = tokens['input_ids'], tokens['attention_mask']


plt.pcolormesh(Xmask)
plt.show()


# #print("Labels Shape")
# #print(labels.shape)

arr=dataset['ec_number_one']
categories = arr.unique().size
labels=np.zeros((arr.size, categories))
labels[np.arange(arr.size), arr-1] = 1


# # Below code is for off loading the data

with open('xids.npy','wb') as f:
    np.save(f,Xids)
with open('xmask.npy','wb') as f:
    np.save(f,Xmask)
with open('labels.npy','wb') as f:
    np.save(f,labels)


# Below code is for load the data

# with open(r'C:\Users\Emre\Desktop\Project\Protein-Active-Site-w-ML\xids.npy','rb') as fp:
#     Xids=np.load(fp)

# with open(r'C:\Users\Emre\Desktop\Project\Protein-Active-Site-w-ML\xmask.npy','rb') as fp:
#     Xmask=np.load(fp)

# with open(r'C:\Users\Emre\Desktop\Project\Protein-Active-Site-w-ML\labels.npy','rb') as fp:
#     labels=np.load(fp)






tensorflow_dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))


def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


tensorflow_dataset = tensorflow_dataset.map(map_func)

# for i in tensorflow_dataset.take(1):
    #print(i)

tensorflow_dataset = tensorflow_dataset.shuffle(100000).batch(BATCH_SIZE)

DS_LEN = len(list(tensorflow_dataset))

#print(DS_LEN)

SPLIT = .9

train = tensorflow_dataset.take(round(DS_LEN * SPLIT))
val = tensorflow_dataset.skip(round(DS_LEN * SPLIT))


bert = TFAutoModel.from_pretrained('Rostlab/prot_bert_bfd')

input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')

embeddings = bert.bert(input_ids, attention_mask=mask)[0]

X = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation='relu')(X) # regularizer here?
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(32, activation='relu')(X)
y = tf.keras.layers.Dense((categories), activation='softmax', name='outputs')(X)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=[y])

model.layers[2].trainable = False


optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])


history = model.fit(
    train,
    validation_data=val,
    epochs=16,
)

model.save("EC_Prediction")
print(history)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
