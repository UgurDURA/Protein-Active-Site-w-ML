
# default variables to use for hyperparameters and settings across all scripts
MAX_LEN = 512       # max length of the input AA sequence
BATCH_SIZE = 16     # Possible Values: 4/8/16/32
DATA_SIZE = 50      # the number of total entries (train + val)

SQLite_DB_PATH = r'/[DATA]/Enzymes.db'
UniProt_XML_PATH = r'/[DATA]/uniprot/uniprot_sprot.xml'

# add to __all__ when you add new variables.
__all__ = ["MAX_LEN", "BATCH_SIZE", "DATA_SIZE", "SQLite_DB_PATH", "UniProt_XML_PATH"]
