'''
TODO: Transformer model tasks
>[check] tokenization
>[check] positional encoding
>[check] token embedding
>[check] combine embeddings
> [] wrap tokenization into a method.
> test detokenization? >sequences_to_text()

> encoding and decoding blocks:
> [] modify encoder block
    > research normalization layer
> [] implement decoder block: fixed-length (4 tokens) output

'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# tokenizers
input_tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters=None, lower=False, char_level=True)
output_tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters=None, lower=False, split='.', char_level=False)

# feed list of integers into fit_on_texts()
# s: example sequence
s = 'MLPPWTLGLLLLATVRGKEVCYGQLGCFSDEKPWAGTLQRPVKLLPWSPEDIDTRFLLYTNENPNNFQLITGTEPDTIEASNFQLDRKTRFIIHGFLDKAEDSWPSDMCKKMFEVEKVNCICVDWRHGSRAMYTQAVQNIRVVGAETAFLIQALSTQLGYSLEDVHVIGHSLGAHTAAEAGRRLGGRVGRITGLDPAGPCFQDEPEEVRLDPSDAVFVDVIHTDSSPIVPSLGFGMSQKVGHLDFFPNGGKEMPGCKKNVLSTITDIDGIWEGIGGFVSCNHLRSFEYYSSSVLNPDGFLGYPCASYDEFQESKCFPCPAEGCPKMGHYADQFKGKTSAVEQTFFLNTGESGNFTSWRYKISVTLSGKEKVNGYIRIALYGSNENSKQYEIFKGSLKPDASHTCAIDVDFNVGKIQKVKFLWNKRGINLSEPKLGASQITVQSGEDGTEYNFCSSDTVEENVLQSLYPC '
ec = '4.3.1.2' #, '3.12.6.-']

input_tok.fit_on_texts(s)
print(s)
s = input_tok.texts_to_sequences(s)
print(s)
vocab = len(input_tok.word_index)
print(vocab)

# ec = ec.replace('.', ' ')
print(ec)
print(type(ec))
output_tok.fit_on_texts(ec)
ec = output_tok.texts_to_sequences(ec)
print(ec)

# def output_tok(m):
#     x = []
#     for num in m:
#         if num != '.' or num != '-':
#             x.append([int(num)])
#         elif num == '-':
#             x.append([1000])
#     return x



embed_dims = 10  # Embedding size for each token
num_heads = 5  # Number of attention heads
ff_dim = 10  # Hidden layer size in feed forward network inside transformer


class embeddings_layer(layers.Layer):
    """
    Positional encoding and token embedding in one layer
    """

    def __init__(self, vocab_size, embed_dim):  # initialize with layer's attributes
        super(embeddings_layer, self).__init__()  # ??
        self.token_emb = layers.Embedding(input_dim=vocab_size + 1, output_dim=embed_dim)
        self.embed_dim = embed_dim
        print("embeddings: init")

    def call(self, x):  # call with the input
        print("embeddings: call")
        length= len(x)
        pos = self.positional_encoding(len(x), self.embed_dim)  # shape: (1, length, mbed_dim)
        x = self.token_emb(tf.cast(x, dtype=tf.int32))          # shape: (lentgh, 1, mbed_dim)

        r = tf.reshape(x, [1, length, self.embed_dim]) + pos
        return r

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

        return tf.cast(pos_encoding, dtype=tf.float32)  # output is in tensor form.


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):  # rate: dropout rate.
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        print("encoder: init")

    def call(self, inputs, training):
        print("encoder: call")
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


imtesting_emb = embeddings_layer(vocab, embed_dims)
x = imtesting_emb(s)
print(x.shape)

imtesting_trn = TransformerBlock(embed_dim=embed_dims, num_heads=num_heads, ff_dim=ff_dim)
x = imtesting_trn(x)

print(x.shape)
