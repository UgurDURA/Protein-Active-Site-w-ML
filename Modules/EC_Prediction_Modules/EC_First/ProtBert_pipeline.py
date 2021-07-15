'''
script to do predictions using the trained models, utilizing the pipeline structure from huggingface transformers.
'''

from transformers import TextClassificationPipeline, AutoTokenizer, TFAutoModelForSequenceClassification

import tensorflow as tf
from transformers import TFAutoModel
MAX_LEN = 256
BATCH_SIZE = 36 # Possible Values: 4/8/16/32

gpu_devices = tf.config.list_physical_devices('GPU')
print(gpu_devices)  # clean up name using strip() maybe.

def main():


    bert = TFAutoModel.from_pretrained('Rostlab/prot_bert_bfd')

    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_ids', dtype='int64')
    mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int64')

    embeddings = bert(input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(16, activation='relu')(X)
    y = tf.keras.layers.Dense(6, activation='softmax', name='outputs')(X)

    bert = tf.keras.Model(inputs=[input_ids, mask], outputs=[y])

    bert.load_weights('./checkpoints/mini_test2/tf_model.h5')

    # TODO: load the tokenizer and the pretrained model from (checkpoints directory)
    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, )
    # bert = TFAutoModelForSequenceClassification.from_pretrained('./checkpoints/mini_test/weights.h5', from_pt=True)
    #
    pipeline = TextClassificationPipeline(model=bert, tokenizer=tokenizer, device=0, framework='tf', task="first EC number prediction")
    # TODO: change device to read from cuda apis

    seq = 'M E N H S K Q T E A P H P G T Y M P A G Y P P P Y P P A A F Q G P S D H A A Y P I P Q A G Y Q G P P G P Y P G P Q P G Y P V P P G G Y A G G ' \
          'G P S G F P V Q N Q P A Y N H P G G P G G T P W M P A P P P P L N C P P G L E Y L A Q I D Q L L V H Q Q I E L L E V L T G F E T N N K Y E I ' \
          'K N S L G Q R V Y F A V E D T D C C T R N C C G A S R P F T L R I L D N L G R E V M T L E R P L R C S S C C F P C C L Q E I E I Q A P P G V ' \
          'P V G Y V T Q T W H P C L P K F T L Q N E K K Q D V L K V V G P C V V C S C C S D I D F E L K S L D E E S V V G K I S K Q W S G F V R E A F ' \
          'T D A D N F G I Q F P L D L D V K M K A V M L G A C F L I D F M F F E R T G N E E Q R S G A W Q '

    print(pipeline(seq))


if __name__ == "__main__":
    main()
