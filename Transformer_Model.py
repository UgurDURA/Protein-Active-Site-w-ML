

import tensorflow as tf
from tensorflow.keras import layers

# tokenizer
tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters=None, lower=False, char_level=True)

# feed list of integers into fit_on_texts()
# s: example sequence
s = 'ANDJIDUVHEJELFJSUDGVKRLRRGMJIFUDN'
tok.fit_on_texts(s)
print(tok.word_index)
vocab_size = len(tok.word_index)

#for every entry
s = tok.texts_to_sequences(s)
print(s)

embed_dim = 10  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 10  # Hidden layer size in feed forward network inside transformer


class embeddings(layers.Layer):
    '''
    Positional and token embedding in one layer
    '''
    def __init__(self, vocab_size, embed_dim):

        #token embedding
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

        #positional encoding, sin cos functions etc.

    def call(self, x):

