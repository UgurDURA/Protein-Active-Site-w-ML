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

    Xids = np.zeros((1,256))
    Xmask = np.zeros((1,256))
    iterable = Requested_sequence  
    separator = " "
    Requested_sequence=separator.join(iterable)


    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, )

    Requested_sequence=re.sub(r"[UZOB]", "X",  Requested_sequence)
    tokens = tokenizer.encode_plus( Requested_sequence, max_length=256, truncation=True, padding="max_length",
                                   add_special_tokens=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='tf')

    Xids[0, :], Xmask[0, :]  = tokens['input_ids'], tokens['attention_mask']

    return Xids, Xmask


print(prepare(Requested_sequence))

# model= tf.keras.models.load_model("")

# prediction=model.predict([prepare()])