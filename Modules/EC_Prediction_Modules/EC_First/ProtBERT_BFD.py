import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import sqlite3

MAX_LEN = 256
BATCH_SIZE = 25 # Possible Values: 4/8/16/32
DATA_SIZE = 500

def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

# dataset = pd.read_csv(r'[DATA]\MainDataset.csv')  # taking data from csv file, you can easily export the data from SQL file to csv
# dataset = dataset.iloc[0:DATA_SIZE, :]

# read data from Enzymes.db, put it into dataset container
con = sqlite3.connect(r'[DATA]\Enzymes.db')

# cur.execute("SELECT ec_number_one, sequence_string FROM EntriesReady LIMIT ('{0}')".format(DATA_SIZE))
# remove LIMIT if you want the entire dataset.
dataset = pd.read_sql_query("SELECT ec_number_one, sequence_string FROM EntriesReady LIMIT ('{0}')".format(DATA_SIZE), con)

# print(dataset.columns)
# Index(['ec_number_one', 'sequence_string'], dtype='object')


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

print(dataset['ec_number_one'].unique)

arr = dataset['ec_number_one'].values

print("Array Size")
print(arr.size)

labels = np.zeros((arr.size, arr.max() + 1))

print("Labels Shape")
print(labels.shape)

labels[np.arange(arr.size), arr] = 1

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

# print("DATASET ON TENSOR FLOW EXAMPLE")
# for i in tensorflow_dataset.take(1):
#     print(i)

tensorflow_dataset = tensorflow_dataset.map(map_func)

for i in tensorflow_dataset.take(1):
    print(i)

tensorflow_dataset = tensorflow_dataset.shuffle(100000).batch(BATCH_SIZE)

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
y = tf.keras.layers.Dense(arr.max() + 1, activation='softmax', name='outputs')(X)

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
    epochs=30
)

print(history)
model.save_weights('./checkpoints/my_checkpoint')
# directory not saved in git. do not forget to clean up the files here and upload to Gdrive with appropriate name when you successfully run a
# training and evaluation loop.
