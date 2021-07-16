import pandas as pd
import sqlite3
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import re
con = sqlite3.connect('[DATA]\Enzymes.db')
cur = con.cursor()

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


MAX_LEN = 512
DATA_SIZE = 10
# LIMIT ('{0}')".format(DATA_SIZE),
dataset = pd.read_sql_query("SELECT Entries.sequence_string, ec_number_one FROM EntriesReady LEFT JOIN Entries WHERE "
                            "EntriesReady.EnzymeAutoID=Entries.EnzymeAutoID AND Entries.sequence_length >'{0}' LIMIT ('{1}')".format(MAX_LEN,
                                                                                                                                     DATA_SIZE), con)

print(dataset)
print(dataset.iloc[0,0])





sequences = []

for e in range(len(dataset)):
    sequences.append(prepare(dataset.iloc[e,0]))


 


arr=dataset['ec_number_one']
categories = arr.unique().size
labels=np.zeros((arr.size, categories))
labels[np.arange(arr.size), arr-1] = 1


print(sequences)
print(labels)
print(categories)


