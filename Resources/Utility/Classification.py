from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_xlnet")
model = AutoModel.from_pretrained("Rostlab/prot_xlnet")


# sequence_Example = "A E T C Z A O"
# sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
# encoded_input = tokenizer(sequence_Example, return_tensors='pt')
# output = model(**encoded_input)
