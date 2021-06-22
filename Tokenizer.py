'''
Tokenizers for input and output. Save the configurations in input_tok_config.txt and output_tok_config.txt as json,
for the purpose of using them to detokenize later.

TODO:
>[check] read from database to train tokenizers
    tokenizers: trained
> save tokenized data to new table in database?

'''

import tensorflow as tf
import json
import sqlite3

con = sqlite3.connect('[DATA]/db/Enzymes.db')
cur = con.cursor()
# connect to database, read 10 entries at a time and process them.

# tokenizers
input_tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters=None, lower=False, char_level=True)
output_tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters='#$%&()+/<=>?@\\^{|}~', lower=False, split='.', char_level=False)

cur.execute('SELECT sequence_string, ec_number_string FROM Entries')
while True:
    batch = cur.fetchmany(10) # batch is list of tuples

    s = [i[0] for i in batch]
    ec = [n[1] for n in batch]

    input_tok.fit_on_texts(s)
    output_tok.fit_on_texts(ec)

    if not batch:
        break
    # each batch contains up to 10 rows


input_config = input_tok.to_json()
output_config = output_tok.to_json()

with open('saves\input_tok_config.txt', 'w') as outfile:
    json.dump(input_config, outfile)

with open('saves\output_tok_config.txt', 'w') as outfile:
    json.dump(output_config, outfile)

con.close()
