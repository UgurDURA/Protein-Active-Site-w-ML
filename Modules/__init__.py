from Modules import Model_Spec, Utility

# default values to test out the models
MAX_LEN = 512       # max length of the input AA sequence
BATCH_SIZE = 16     # Possible Values: 4/8/16/32
DATA_SIZE = 50      # the number of total entries (train + val)
SQLite_DB_PATH = r'[DATA]/Enzymes.db'
UniProt_XML_PATH = r'[DATA]/uniprot/uniprot_sprot.xml'

__all__ = ["Model_Spec", "Utility", MAX_LEN, BATCH_SIZE, DATA_SIZE, SQLite_DB_PATH, UniProt_XML_PATH]
