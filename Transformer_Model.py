'''
TODO: Transformer model tasks
>[check] input & output tokenization
>[check] positional encoding
>[check] token embedding
>[check] combine embeddings
>[check] test detokenization -> sequences_to_text()

> [check] modify encoder block
> [] classification layer(s) to fit output format - 4 tokens: 4 classifctn layers?
> [] decide and implement evaluation criteria

data preprocessing steps:
> [check] tokenizer training in separate script.
> store tokenized data?
> [] method to divide/save data as train-validtn


https://keras.io/api/models/model_training_apis/
https://keras.io/guides/serialization_and_saving/
'''

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers


# feed list of strings into fit_on_texts()
# s: example sequence
s = ['MLPPWTLGLLLLATVRGKEVCYGQLGCFSDEKPWAGTLQRPVKLLPWSPEDIDTRFLLYTNENPNNFQLITGTEPDTIEASNFQLDRKTRFIIHGFLDKAEDSWPSDMCKKMFEVEKVNCICVDWRHGSRAMYTQAVQNIRVVGAETAFLIQALSTQLGYSLEDVHVIGHSLGAHTAAEAGRRLGGRVGRITGLDPAGPCFQDEPEEVRLDPSDAVFVDVIHTDSSPIVPSLGFGMSQKVGHLDFFPNGGKEMPGCKKNVLSTITDIDGIWEGIGGFVSCNHLRSFEYYSSSVLNPDGFLGYPCASYDEFQESKCFPCPAEGCPKMGHYADQFKGKTSAVEQTFFLNTGESGNFTSWRYKISVTLSGKEKVNGYIRIALYGSNENSKQYEIFKGSLKPDASHTCAIDVDFNVGKIQKVKFLWNKRGINLSEPKLGASQITVQSGEDGTEYNFCSSDTVEENVLQSLYPC ',
'MIGRLNHVAIAVPDLEAAAAQYRNTLGAEVGAPQDEPDHGVTVIFITLPNTKIELLHPLGEGSPIAGFLEKNPAGGIHHICYEVEDILAARDRLKEAGARVLGSGEPKIGAHGKPVLFLHPKDFNGCLVELEQV']
ec = ['4.1.1.15', '5.1.99.-']


# tokenizers
input_tok, output_tok = [], []

with open('input_tok_config.txt') as json_file:
    config = json.load(json_file)
    input_tok = tf.keras.preprocessing.text.tokenizer_from_json(config)

with open('output_tok_config.txt') as json_file:
    config = json.load(json_file)
    output_tok = tf.keras.preprocessing.text.tokenizer_from_json(config)

s = input_tok.texts_to_sequences(s)
ec = output_tok.texts_to_sequences(ec)
vocab = len(input_tok.word_index)


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
        # self.dropout1 = layers.Dropout(rate)
        # self.dropout2 = layers.Dropout(rate)
        print("encoder: init")

    def call(self, inputs, training):
        print("encoder: call")
        attn_output = self.att(inputs, inputs)
        # attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

inputs = layers.Input()     # input layer of unspecified dimensions
imtesting_emb = embeddings_layer(vocab, embed_dims)
x = imtesting_emb(inputs)
imtesting_trn = TransformerBlock(embed_dim=embed_dims, num_heads=num_heads, ff_dim=ff_dim)
x = imtesting_trn(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
)


