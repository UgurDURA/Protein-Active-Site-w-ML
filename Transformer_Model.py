import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import xml_parse_methods
import os
import xml.etree.ElementTree as ET

file_name = 'ExampleDATA.xml'
full_file = os.path.abspath(os.path.join(file_name))


def main():


    print('That\'s all folks!')
