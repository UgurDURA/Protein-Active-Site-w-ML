import Modules

# default values to test out the models

MAX_LEN = 512       # max length of the input AA sequence
BATCH_SIZE = 16     # Possible Values: 4/8/16/32
DATA_SIZE = 50      # the number of total entries (train + val)

__all__ = ["Modules", MAX_LEN, BATCH_SIZE, DATA_SIZE]
