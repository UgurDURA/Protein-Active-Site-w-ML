import logging
# import numpy as np
import pandas as pd
# import re
import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetForSequenceClassification, TFTrainer, TFTrainingArguments
# from datasets import load_dataset
from sklearn.model_selection import train_test_split


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


# figure out the dataset and splitting
# preprocess:
# sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)

# pull data from ExampleDATA.csv, preprocess the input sequences like above,
# preprocess the ec numbers to be one of 7 categories.

# dataset = load_dataset('csv', data_files=['[DATA]/DummyData/ExampleDataReady.csv'])
# alternatively: use pandas? read from db?
# shape : (90, 5)
# dict_values([Dataset({
#     features: ['EnzymeAutoID', 'accession_string', 'ec_number_string', 'sequence_length', 'sequence_string'],
#     num_rows: 90
# })])


dataset = pd.read_csv('../../../[DATA]/DummyData/ExampleDataReady.csv')
EcNumberDataset = list(dataset.iloc[:, 2])  # features         : ec_number_string
SequenceDataset = list(dataset.iloc[:, 6])  # Dependent values : sequence_string


tokenizer = XLNetTokenizer.from_pretrained("../../../Resources/Models/prot_xlnet")

SequenceDataset = tokenizer(SequenceDataset, return_tensors="tf")       # input ID's aka tokenized sequences.

# prune head and add new classifier
XLNet_model = TFXLNetForSequenceClassification.from_pretrained("../../../Resources/Models/prot_xlnet",
                                                               config="../../../Resources/Models/prot_xlnet/config.json", from_pt=True)
inputs = tf.keras.layers.Input()
embeddings = XLNet_model(inputs)[0]
pooling = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
norm = tf.keras.layers.BatchNormalization()(pooling)
hidden64 = tf.keras.layers.Dense(64, activation='relu')(norm)
drop = tf.keras.layers.Dropout(0.1)(hidden64)
hidden32 = tf.keras.layers.Dense(32, activation='relu')(drop)
classification = tf.keras.layers.Dense(8, activation='softmax', name='outputs')(hidden32)

opt = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')


tensorflow_dataset = tf.data.Dataset.from_tensor_slices((SequenceDataset, EcNumberDataset))
tensorflow_dataset = tensorflow_dataset.shuffle(100000).batch(32)
DS_LEN = len(list(tensorflow_dataset))
SPLIT = .9
train = tensorflow_dataset.take(round(DS_LEN*SPLIT))
val = tensorflow_dataset.skip(round(DS_LEN*SPLIT))


model = tf.keras.Model(inputs=[], outputs=[classification])
model.layers[0].trainable = False
model.summary()

model.compile(optimizer=opt, loss=loss, metrics=[acc])

history = model.fit(
    train,
    validation_data=val,
    epochs=100,
)

print(history)


# training_args = TFTrainingArguments(
#     output_dir='./Results/XLNet',          # output directory
#     num_train_epochs=30,              # total number of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./results/XLNet/logs',            # directory for storing logs
#     logging_steps=100,
# )
#
# trainer = TFTrainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=val_dataset             # evaluation dataset
# )
#
# trainer.train()

