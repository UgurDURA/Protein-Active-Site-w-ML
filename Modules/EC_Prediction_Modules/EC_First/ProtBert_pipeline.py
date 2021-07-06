import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import sqlite3


'''
TODO:
import or copy input preprocessing method/s
set up pipeline

possible new models to use and compare:
ProtT5-XXL-BFD: t5 is the original transformer architecture, it is the most succesful model from ProtTrans, but it is 4 times the size of ProtBERT 
in  memory.
Prot_XLNet: no limit in sequence length? but input layer needs to be figured out in order to use it.
'''

'''
immediate todo:
configure proj to use files in imports
implement and test out pipeline for BERT-BFD
try out trainer from huggingface
fix Prot_xlnet into working
'''