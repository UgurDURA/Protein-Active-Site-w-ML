from enum import unique
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf

import os
import transformers
import matplotlib.pyplot as plt
import re

print("-----------------------------Welcome to ECOPRO EC Number Prediction-------------------")

Requested_sequence=input('Please request a sequence: ')



def prepare(Requested_sequence):

    iterable = Requested_sequence  
    separator = " "
    Requested_sequence=separator.join(iterable)


    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, )

    Requested_sequence=re.sub(r"[UZOB]", "X",  Requested_sequence)
    tokens = tokenizer(Requested_sequence, max_length=512, truncation=True, padding="max_length",
                                   add_special_tokens=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='tf')


    return {
        'input_ids': tf.cast(tokens['input_ids'], tf.float64),
        'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
    }



test=prepare(Requested_sequence)

model = tf.keras.models.model_from_json('results/config.json')
model.load_weights('results/tf_model.h5')

model.summary()

result=model.predict(test)

print(result)

result=np.argmax(result[0])

print(result)
