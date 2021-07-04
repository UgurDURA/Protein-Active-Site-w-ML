import logging
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetForSequenceClassification, TFTrainer, TFTrainingArguments
from datasets import load_dataset


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


# figure out the dataset and splitting
# preprocess:
# sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)

# pull data from ExampleDATA.csv, preprocess the input sequences like above,
# preprocess the ec numbers to be one of 7 categories.

dataset = load_dataset('csv', data_files=['[DATA]/DummyData/ExampleDATA.csv'])
# alternatively: use pandas? read from db?
# shape : (90, 5)
# dict_values([Dataset({
#     features: ['EnzymeAutoID', 'accession_string', 'ec_number_string', 'sequence_length', 'sequence_string'],
#     num_rows: 90
# })])




tokenizer = XLNetTokenizer.from_pretrained("Resources/Models/prot_xlnet")

# prune head and add new classifier
XLNet_model = TFXLNetForSequenceClassification.from_pretrained("Resources/Models/prot_xlnet",
                                                         config="Resources/Models/prot_xlnet/config.json", from_pt=True)

embeddings = XLNet_model()[0]
pooling = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
norm = tf.keras.layers.BatchNormalization()(pooling)
hidden128 = tf.keras.layers.Dense(128, activation='relu')(norm)
drop = tf.keras.layers.Dropout(0.1)(hidden128)
hidden32 = tf.keras.layers.Dense(32, activation='relu')(drop)
classification = tf.keras.layers.Dense(8, activation='softmax', name='outputs')(hidden32)

opt = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model = tf.keras.Model(inputs=[], outputs=[classification])
model.layers[0].trainable = False
model.summary()

model.compile(optimizer=opt, loss=loss, metrics=[acc])


training_args = TFTrainingArguments(
    output_dir='./Results/XLNet',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./results/XLNet/logs',            # directory for storing logs
    logging_steps=100,
)

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# trainer.train()

