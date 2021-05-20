'''
TODO: Transformer model tasks
>[check] tokenization
>[check] positional encoding
>[] token embedding
>[] combine embeddings

> encoding and decoding blocks

'''

import numpy as np
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

# for every entry
s = tok.texts_to_sequences(s)  # taking the string and turning it into a sequence of integers
print(s)
# print( tok.detokenize?(s) )

embed_dim = 10  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 10  # Hidden layer size in feed forward network inside transformer


class embeddings(layers.Layer):
    """
    Positional and token embedding in one layer
    """

    def __init__(self, vocab_size, embed_dim):
        # token embedding here?
        # tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters=None, lower=False, char_level=True)
        print("init")
        # self.token_enc = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters=None, lower=False, char_level=True)
        # self.pos_emb = layers.Embedding( )

    def call(self, x):
        print("call")

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):  # position: as in string length. d_model: dimensions
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)      #output is in tensor form.

    def token_embedding(self, ):
        print("token embeddings")
        # fill this up


imtesting = embeddings(tok.num_words, embed_dim)
print(imtesting.positional_encoding(10, 5))