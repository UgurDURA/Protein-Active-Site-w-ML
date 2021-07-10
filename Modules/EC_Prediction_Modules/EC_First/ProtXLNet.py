import logging
import sqlite3
import pandas as pd
import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetForSequenceClassification, TFTrainer, TFTrainingArguments
from datasets import Dataset

MAX_LEN = 512
BATCH_SIZE = 16  # Possible Values: 4/8/16/32
DATA_SIZE = 5000

def main():
    logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

    con = sqlite3.connect(r'..\..\..\[DATA]\Enzymes.db')
    dataset = pd.read_sql_query("SELECT ec_number_one, sequence_string FROM EntriesReady LIMIT ('{0}')".format(DATA_SIZE), con)

    tokenizer = XLNetTokenizer.from_pretrained("../../../Resources/Models/prot_xlnet")

    SequenceDataset = []
    for i, sequence in enumerate(dataset['sequence_string']):
        SequenceDataset.append(tokenizer(sequence, add_special_tokens=True, return_token_type_ids=False, return_tensors='tf'))
    tf.ragged.constant(SequenceDataset)

    EcNumberDataset = dataset['ec_number_one']
    labels = dataset['ec_number_one'].values

    tensorflow_dataset = tf.data.Dataset.from_tensor_slices((SequenceDataset, EcNumberDataset))
    tensorflow_dataset = tensorflow_dataset.shuffle(100000).batch(BATCH_SIZE)
    DS_LEN = len(list(tensorflow_dataset))
    print('whole dataset length: ' + str(DS_LEN))
    SPLIT = .9
    train = tensorflow_dataset.take(round(DS_LEN * SPLIT))
    val = tensorflow_dataset.skip(round(DS_LEN * SPLIT))

    # prune head and add new classifier
    XLNet_model = TFXLNetForSequenceClassification.from_pretrained("../../../Resources/Models/prot_xlnet",
                                                                   config="../../../Resources/Models/prot_xlnet/config.json", from_pt=True)
    # TODO : edit this layer properly
    inputs = tf.keras.layers.InputLayer(input_shape=None, name="input layer", ragged=True, type_spec=None)

    embeddings = XLNet_model(inputs)[0]
    X = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(16, activation='relu')(X)
    y = tf.keras.layers.Dense(labels.max() + 1, activation='softmax', name='outputs')(X)

    model = tf.keras.Model(inputs=[inputs], outputs=[y])

    model.layers[1].trainable = False
    model.summary()

    optimizer = tf.keras.optimizers.Adam(0.01)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    history = model.fit(
        train,
        validation_data=val,
        epochs=15,
    )

    # print(history)
    # model.save_weights('./checkpoints/my_checkpoint')  # change checkpoint dir/name before running
    # # directory not saved in git. do not forget to clean up the files here and upload to Gdrive with appropriate name when you successfully run a
    # # training and evaluation loop.

    # training_args = TFTrainingArguments(
    #     output_dir='./Results/XLNet',          # output directory
    #     num_train_epochs=30,              # total number of training epochs
    #     per_device_train_batch_size=16,  # batch size per device during training
    #     per_device_eval_batch_size=64,   # batch size for evaluation
    #     warmup_steps=500,                # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,               # strength of weight decay
    #     logging_dir='./results/XLNet/logs',            # directory for storing logs
    #     logging_steps=100,
    # )
    #
    # trainer = TFTrainer(
    #     model=model,                         # the instantiated 🤗 Transformers model to be trained
    #     args=training_args,                  # training arguments, defined above
    #     train_dataset=train_dataset,         # training dataset
    #     eval_dataset=val_dataset             # evaluation dataset
    # )
    #
    # trainer.train()

if __name__ == "__main__":
    main()