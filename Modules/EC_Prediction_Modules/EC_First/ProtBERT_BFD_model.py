'''
EC Number classification using ProtBERT pretrained model as embeddings.
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

MAX_LEN = 256

tokenizer = AutoTokenizer.from_pretrained('../../../Resources/Models/prot_bert_bfd', do_lower_case=False, )
# tokens = tokenizer.encode_plus(Sequence_Example, max_length=MAX_LEN, truncation=True, padding="max_length",
#                                add_special_tokens=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='tf')

dataset = pd.read_csv('../../../[DATA]/DummyData/ExampleDataReady.csv')
# EcNumberDataset = list(dataset.iloc[:, 2])  # features         : ec_number_one
# SequenceDataset = list(dataset.iloc[:, 6])  # Dependent values : sequence_string

Xids = np.zeros((len(dataset), MAX_LEN))
Xmask = np.zeros((len(dataset), MAX_LEN))

print("XIDS SHAPE")
print(Xids.shape)

for i, sequence in enumerate(dataset['sequence_string'].values):
    tokens = tokenizer(sequence, max_length=MAX_LEN, truncation=True, padding="max_length",
                       add_special_tokens=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='tf')

    Xids[i, :], Xmask[i, :] = tokens['input_ids'], tokens['attention_mask']

# print("XIDS")
# print(type(Xids))
# print("XMASKS")
# print(Xmask)

print(dataset['ec_number_one'].unique)
arr = dataset['ec_number_one'].values

print("Array Size")
print(arr.size)

labels = np.zeros((arr.size, arr.max() + 1))

print("Labels Shape")
print(labels.shape)

labels[np.arange(arr.size), arr] = 1

del dataset

# print("LABELS")
# print(labels)

# # Below code is for off loading the data
# with open('../../../Resources/ProtBERT_dataset_saves/xids.npy', 'wb+') as f:
#     np.save(f, Xids)
# with open('../../../Resources/ProtBERT_dataset_saves/xmask.npy', 'wb+') as f:
#     np.save(f, Xmask)
# with open('../../../Resources/ProtBERT_dataset_saves/labels.npy', 'wb+') as f:
#     np.save(f, labels)

# Below code is for load the data
# with open('xids.npy','rb') as fp:
#     Xids=np.load(fp)

# with open('xmask.npy','rb') as fp:
#     Xmask=np.load(fp)

# with open('labels.npy','rb') as fp:
#     labels=np.load(fp)


tf.config.experimental.list_physical_devices('GPU')

tensorflow_dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))


# print("DATASET ON TENSOR FLOW EXAMPLE")
# for i in tensorflow_dataset.take(1):
#     print(i)


def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


tensorflow_dataset = tensorflow_dataset.map(map_func)
tensorflow_dataset = tensorflow_dataset.shuffle(100000).batch(8)

DS_LEN = len(list(tensorflow_dataset))

print(DS_LEN)

SPLIT = .9

train = tensorflow_dataset.take(round(DS_LEN * SPLIT))
val = tensorflow_dataset.skip(round(DS_LEN * SPLIT))

del tensorflow_dataset

bert = TFAutoModel.from_pretrained('../../../Resources/Models/prot_bert_bfd', config='../../../Resources/Models/prot_bert_bfd/config.json')

input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')

embeddings = bert(input_ids, attention_mask=mask)[0]

X = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(32, activation='relu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(16, activation='relu')(X)
y = tf.keras.layers.Dense(7, activation='softmax', name='outputs')(X)

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
    epochs=5,
)

print(history)
