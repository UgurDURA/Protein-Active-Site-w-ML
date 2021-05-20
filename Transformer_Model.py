'''
TODO: Transformer model tasks
>[check] tokenization
> [] wrap tokenization into a method.
>[check] positional encoding
>[check] token embedding
>[] combine embeddings
> test detokenization?

> encoding and decoding blocks

'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# tokenizer
tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters=None, lower=False, char_level=True)

# feed list of integers into fit_on_texts()
# s: example sequence
s = 'MLPPWTLGLLLLATVRGKEVCYGQLGCFSDEKPWAGTLQRPVKLLPWSPEDIDTRFLLYTNENPNNFQLITGTEPDTIEASNFQLDRKTRFIIHGFLDKAEDSWPSDMCKKMFEVEKVNCICVDWRHGSRAMYTQAVQNIRVVGAETAFLIQALSTQLGYSLEDVHVIGHSLGAHTAAEAGRRLGGRVGRITGLDPAGPCFQDEPEEVRLDPSDAVFVDVIHTDSSPIVPSLGFGMSQKVGHLDFFPNGGKEMPGCKKNVLSTITDIDGIWEGIGGFVSCNHLRSFEYYSSSVLNPDGFLGYPCASYDEFQESKCFPCPAEGCPKMGHYADQFKGKTSAVEQTFFLNTGESGNFTSWRYKISVTLSGKEKVNGYIRIALYGSNENSKQYEIFKGSLKPDASHTCAIDVDFNVGKIQKVKFLWNKRGINLSEPKLGASQITVQSGEDGTEYNFCSSDTVEENVLQSLYPC '

tok.fit_on_texts(s)
print(s)
s = tok.texts_to_sequences(s)
print(s)
vocab = len(tok.word_index)
print(vocab)

embed_dims = 10  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 10  # Hidden layer size in feed forward network inside transformer


class embeddings_layer(layers.Layer):
    """
    Positional encoding and token embedding in one layer
    """

    def __init__(self, vocab_size, embed_dim):  # initialize with layer's attributes
        super(embeddings_layer, self).__init__()  # ??
        self.token_emb = layers.Embedding(input_dim=vocab_size+1, output_dim=embed_dim)
        self.embed_dim = embed_dim
        print("init")

    def call(self, x):  # call with the input

        pos = self.positional_encoding(len(x), self.embed_dim)
        x = self.token_emb(tf.cast(x, dtype=tf.int32))
        print("call")

        return x + pos

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


imtesting = embeddings_layer(vocab, embed_dims)

print(imtesting(s))