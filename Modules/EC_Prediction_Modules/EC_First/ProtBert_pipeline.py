'''
script to do predictions using the trained models, utilizing the pipeline structure from huggingface transformers.
'''
import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
from transformers import TextClassificationPipeline, AutoTokenizer, TFAutoModelForSequenceClassification
from Modules.Utility.DataManipulation import addSpaces


def main():
    # TODO: load the tokenizer and the pretrained model from (checkpoints directory)

    tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd_localization', do_lower_case=False, )
    bert = TFAutoModelForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd_localization', from_pt=True)

    pipeline = TextClassificationPipeline(task="text-classification", model=bert, tokenizer=tokenizer, device=0, framework='tf', )
    # TODO: change device to read from cuda api

    seq = 'M E N H S K Q T E A P H P G T Y M P A G Y P P P Y P P A A F Q G P S D H A A Y P I P Q A G Y Q G P P G P Y P G P Q P G Y P V P P G G Y A G G ' \
          'G P S G F P V Q N Q P A Y N H P G G P G G T P W M P A P P P P L N C P P G L E Y L A Q I D Q L L V H Q Q I E L L E V L T G F E T N N K Y E I ' \
          'K N S L G Q R V Y F A V E D T D C C T R N C C G A S R P F T L R I L D N L G R E V M T L E R P L R C S S C C F P C C L Q E I E I Q A P P G V ' \
          'P V G Y V T Q T W H P C L P K F T L Q N E K K Q D V L K V V G P C V V C S C C S D I D F E L K S L D E E S V V G K I S K Q W S G F V R E A F ' \
          'T D A D N F G I Q F P L D L D V K M K A V M L G A C F L I D F M F F E R T G N E E Q R S G A W Q '

    print(pipeline(seq))

    # TODO: add func to format the output of pipeline.
