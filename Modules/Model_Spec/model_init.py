from transformers import AutoTokenizer, TFAutoModel, TFBertModel, XLNetTokenizer, TFXLNetForSequenceClassification
import tensorflow as tf
from ... import *

def create_model(n_dense1=64, n_dense2=16,dout_rate=0.1, ** kwargs):
    embedding_base = kwargs.embedding_base      # specify ProtBERT_BFD or XLNET
    categories = kwargs.categories              # number of labels

    # acrobatics to avoid putting a model inside a model in keras which prevents saving the model
    if embedding_base == "ProtBERT_BFD":
        base = TFAutoModel.from_pretrained('Rostlab/prot_bert_bfd')
        assert isinstance(base, TFBertModel)
        main_layer = base.bert

        input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')

        embeddings = main_layer(input_ids, attention_mask=mask)[0]

    elif embedding_base == "XLNET":     # TODO: probably needs debugging
        base = TFAutoModel.from_pretrained("Rostlab/prot_xlnet", from_pt=True)
        assert isinstance(base, TFXLNetForSequenceClassification)
        main_layer = base.xlnet

        inputs = tf.keras.layers.Input(shape=None, name="input layer", ragged=True, type_spec=None)
        embeddings = main_layer(inputs)[0]

    else:
        print("create_model(): invalid arg")
        # throw error
    del base

    # TODO: fix input tensor issue from embedding layers : [0]
    X = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(n_dense1, activation='relu')(X)
    X = tf.keras.layers.Dropout(dout_rate)(X)
    X = tf.keras.layers.Dense(n_dense2, activation='relu')(X)
    y = tf.keras.layers.Dense(categories, activation='softmax', name='outputs')(X)
    # if you are going to adjust the inner workings of the classification head, do so here.

    model = tf.keras.Model(inputs=(input_ids, mask), outputs=[y])
    model.layers[2].trainable = False

    return model

def create_tokenizer(model):
    if model == "ProtBERT_BFD":
        tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    elif model == "XLNET":
        tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
    else:
        print("create_model(): invalid arg")
        # throw error
    return tokenizer
