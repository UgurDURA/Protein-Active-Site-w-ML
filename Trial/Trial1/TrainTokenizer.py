#Install r-tokenizers and r-tokenizers.bpe on conda environment

from pathlib import Path
import sqlite3
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# con = sqlite3.connect('[DATA]/db/Enzymes.db')
# cur = con.cursor()


# cur.execute('SELECT sequence_string, ec_number_string FROM Entries')

# f = open("Sequences.txt", "w")
# f = open("Ec_Numbers.txt", "w")

# while True:


#     batch = cur.fetchmany(10) # batch is list of tuples
#     # Customize training

#     s = [i[0] for i in batch]
#     ec = [n[1] for n in batch]

#     SequenceFile = open("Sequences.txt", "a")
#     for element in s:
#         SequenceFile.write(element + "\n")
#     SequenceFile.close()

#     EcNumberFile = open("Ec_Numbers.txt", "a")
#     for element in ec:
#         EcNumberFile.write(element + "\n")
#     EcNumberFile.close()


#     if not batch:
#         break
    
    
# paths = [str(x) for x in Path("[DATA]\DB\Sequences.txt").glob("**/*.txt")] 

# tokenizer = ByteLevelBPETokenizer() # Initialize a tokenizer   
    
# tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
#         "<s>",
#         "<pad>",
#         "</s>",
#         "<unk>",
#         "<mask>",
#     ])

# # Save files to disk
# tokenizer.save_model(".", "SequenceTokens")

# con.close()


tokenizer = ByteLevelBPETokenizer(
    "Models\SequenceTokens-vocab.json",
    "Models\SequenceTokens-merges.txt",
)
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)

print(tokenizer.encode("LLQPLVFSSVLSWIPQTGVIFINLVVCWSYYAYVVELCI")

)
