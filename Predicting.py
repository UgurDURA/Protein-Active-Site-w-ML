from enum import unique
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

import os
import transformers
import matplotlib.pyplot as plt
import re
import sqlite3

MAX_LEN = 512
DATA_SIZE = 50


# print("-----------------------------Welcome to ECOPRO EC Number Prediction-------------------")
#
# Requested_sequence=input('Please request a sequence: ')


def prepare(Requested_sequence):

    iterable = Requested_sequence  
    separator = " "
    Requested_sequence=separator.join(iterable)


    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, )

    Requested_sequence=re.sub(r"[UZOB]", "X",  Requested_sequence)
    tokens = tokenizer(Requested_sequence, max_length=512, truncation=True, padding="max_length",
                                   add_special_tokens=False, return_token_type_ids=False, return_attention_mask=True, return_tensors='tf')


    return {
        'input_ids': tf.cast(tokens['input_ids'], tf.float64),
        'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
    }


con = sqlite3.connect('[DATA]\Enzymes.db')
cur = con.cursor()

# LIMIT ('{0}')".format(DATA_SIZE),
dataset = pd.read_sql_query("SELECT sequence_string, ec_number_one FROM EntriesReady LEFT JOIN Entries WHERE "
                            "EntriesReady.EnzymeAutoID=Entries.EnzymeAutoID AND Entries.sequence_length >'{0}' LIMIT ('{1}')".format(MAX_LEN,
                                                                                                                                     DATA_SIZE), con)


sequences = []
ecnums = []
for e in dataset:
    sequences.append(prepare(e['sequence_string']))
    ecnums.append(e['ec_number_one'])

labels=np.zeros((ecnums.size, 7))
labels[np.arange(ecnums.size), ecnums-1] = 1


model = tf.keras.models.load_model("EC_Prediction")
# model.load_weights('results/tf_model.h5')

# model.summary()

# result=model.predict_on_batch()   # to get prediction probability values
result=model.test_on_batch(sequences, ecnums)      # to get metric score


# result=np.argmax(result[0])
print(result)
