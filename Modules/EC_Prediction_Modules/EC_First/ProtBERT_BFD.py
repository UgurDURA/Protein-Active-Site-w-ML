import sys

sys.path.append('../../')
from Modules.Utility.DataManipulation import map_func
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel, TFTrainingArguments, TFTrainer

import tensorflow as tf
import sqlite3

MAX_LEN = 512
BATCH_SIZE = 16  # Possible Values: 4/8/16/32
DATA_SIZE = 50


# TODO: configure here to read args from cmmndline, set defaults
def main():
    # read data from Enzymes.db, put it into dataset container
    con = sqlite3.connect(r'..\..\..\[DATA]\Enzymes.db')

    # remove LIMIT if you want the entire dataset.
    dataset = pd.read_sql_query("SELECT ec_number_one, sequence_string FROM EntriesReady LIMIT ('{0}')".format(DATA_SIZE), con)
    # DataFrame object holding our dataset.

    print('eager execution: ', tf.executing_eagerly())

    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, )

    Xids = np.zeros((len(dataset), MAX_LEN))
    Xmask = np.zeros((len(dataset), MAX_LEN))
    #
    # print("XIDS SHAPE")
    # print(Xids.shape)

    # print(dataset['sequence_string'])

    tokens = []
    for i, sequence in enumerate(dataset['sequence_string']):
        tokens = (tokenizer(sequence[1], max_length=MAX_LEN, truncation=True, padding="max_length", add_special_tokens=True,
                                return_token_type_ids=False, return_attention_mask=True, return_tensors='tf'))
        Xids[i, :], Xmask[i, :] = tokens['input_ids'], tokens['attention_mask']

    print("XIDS: ", type(Xids))
    print("XMASKS: ", type(Xmask))

    ecnumbers = dataset['ec_number_one']

    print("unique ec numbers: ", ecnumbers.unique())
    arr = ecnumbers.values

    print("Array Size: ", arr.size)

    labels = np.zeros((arr.size, arr.max() + 1))

    print("Labels Shape", labels.shape)

    labels[np.arange(arr.size), arr] = 1  #effectively one hot encoding.

    # print("LABELS")
    # print(labels)

    # # Below code is for off loading the data

    # with open('xids.npy','wb') as f:
    #     np.save(f,Xids)
    # with open('xmask.npy','wb') as f:
    #     np.save(f,Xmask)
    # with open('labels.npy','wb') as f:
    #     np.save(f,labels)

    # Below code is for load the data

    # with open(r'C:\Users\ugur_\Desktop\Projects\Protein-Active-Site-w-ML\xids.npy','rb') as fp:
    #     Xids=np.load(fp)

    # with open(r'C:\Users\ugur_\Desktop\Projects\Protein-Active-Site-w-ML\xmask.npy','rb') as fp:
    #     Xmask=np.load(fp)

    # with open(r'C:\Users\ugur_\Desktop\Projects\Protein-Active-Site-w-ML\labels.npy','rb') as fp:
    #     labels=np.load(fp)

    tensorflow_dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

    # print("DATASET ON TENSOR FLOW EXAMPLE")
    # for i in tensorflow_dataset.take(1):
    #     print(i)
    DS_LEN = int(tensorflow_dataset.__len__())  # cast as int because returns EagerTensor for some reason
    print("dataset length: ", DS_LEN)
    SPLIT = .9

    tensorflow_dataset = tensorflow_dataset.map(map_func)

    # for i in tensorflow_dataset.take(1):
    #     print(i)

    tensorflow_dataset = tensorflow_dataset.shuffle(42).batch(BATCH_SIZE)
    train = tensorflow_dataset.take(round(DS_LEN * SPLIT))
    val = tensorflow_dataset.skip(round(DS_LEN * SPLIT))

    # all above code is not working as we want it to.
    # TODO: custom huggingface Dataset

    bert = TFAutoModel.from_pretrained('Rostlab/prot_bert_bfd')

    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')

    embeddings = bert(input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(16, activation='relu')(X)
    y = tf.keras.layers.Dense(arr.max() + 1, activation='softmax', name='outputs')(X)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=[y])

    model.layers[2].trainable = False
    model.summary()

    optimizer = tf.keras.optimizers.Adam(0.01)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    history = model.fit(
        train,
        validation_data=val,
        epochs=10
    )

    print(history)

    # important problem: throwing NotImplementedError
    json_config = model.get_config()
    # print this json to file
    print(json_config)
    model.save('./checkpoints/mini_test2/tf_model.h5py')

    tf.keras.models.save_model(
        model, './checkpoints/mini_test2/tf_model.h5', overwrite=True,  save_format='h5',
        save_traces=True)

    # directory not saved in git. do not forget to clean up the files here and upload to Gdrive with appropriate name when you successfully run a
    # training and evaluation loop.

    #TODO: fix slicing of the dataset and use trainer API  instead.

    # training_args = TFTrainingArguments(
    #     output_dir='./Results/ProtBERT',  # output directory
    #     overwrite_output_dir=True,
    #     num_train_epochs=10,  # total number of training epochs
    #     per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
    #     per_device_eval_batch_size=BATCH_SIZE * 2,  # batch size for evaluation
    #     # warmup_steps=50,                # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,  # strength of weight decay
    #     logging_dir='./results/ProtBERT/logs',  # directory for storing logs
    #     logging_steps=100,
    # )
    #
    # trainer = TFTrainer(
    #     model=model,  # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,  # training arguments, defined above
    #     train_dataset=train,  # training dataset
    #     eval_dataset=val  # evaluation dataset
    # )
    #
    # trainer.train()


if __name__ == "__main__":
    main()
