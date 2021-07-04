import logging
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetForSequenceClassification, TFTrainer, TFTrainingArguments

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

tokenizer = XLNetTokenizer.from_pretrained("Resources/Models/prot_xlnet")

# prune head and add new classifier
model = TFXLNetForSequenceClassification.from_pretrained("Resources/Models/prot_xlnet", config="Resources/Models/prot_xlnet/config.json", from_pt=True)


# figure out the dataset and splitting
# preprocess:
# sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)

# pull data from ExampleDATA.csv, preprocess the input sequences like above,
# preprocess the ec numbers to be one of 7 categories.





training_args = TFTrainingArguments(
    output_dir='./results/XLNet',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./results/logs',            # directory for storing logs
    logging_steps=10,
)

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# trainer.train()

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss) # can also use any keras loss fn
# model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)
