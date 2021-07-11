from pathlib import Path
import os


# ROOT_DIR = Path(__file__).parent.parent.parent  # This is your Project Root
# SQLite_DB_PATH = str(os.path.join(ROOT_DIR, r'/[DATA]/Enzymes.db'))
# UniProt_XML_PATH = os.path.join(ROOT_DIR, r'/[DATA]/uniprot/uniprot_sprot.xml')


SQLite_DB_PATH = os.path.abspath(r'../../[DATA]/Enzymes.db')
UniProt_XML_PATH = os.path.abspath(r'../../[DATA]/uniprot/uniprot_sprot.xml')

print(SQLite_DB_PATH)
