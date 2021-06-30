#Install r-tokenizers and r-tokenizers.bpe on conda environment
from torch.utils.data import Dataset
from pathlib import Path
import sqlite3
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

con = sqlite3.connect('[DATA]/db/Enzymes.db')
cur = con.cursor()


cur.execute('SELECT sequence_string, ec_number_string FROM Entries')

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


# tokenizer = ByteLevelBPETokenizer(
#     "Models\SequenceTokens-vocab.json",
#     "Models\SequenceTokens-merges.txt",
# )
# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )

# print(tokenizer.encode("LLQPLVFSSVLSWIPQTGVIFINLVVCWSYYAYVVELCI")

# )

class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            "Models\SequenceTokens-vocab.json",
            "Models\SequenceTokens-merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=35000)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = Path("[DATA]\TestData\SequenceTest.txt").glob("*-eval.txt") if evaluate else Path("[DATA]\TrainingData\SequencesTraining.txt").glob("*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./models/EsperBERTo-small",
    tokenizer="./models/EsperBERTo-small"
)

result = fill_mask("La suno <mask>.")


print('Hello world') 