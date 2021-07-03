import re
from transformers import XLNetTokenizer, TFXLNetModel

tokenizer = XLNetTokenizer.from_pretrained("Resources/Models/prot_xlnet")

model = TFXLNetModel.from_pretrained("Resources/Models/prot_xlnet", config="Resources/Models/prot_xlnet/config.json", from_pt=True)

# , from_pt=True

# example sequence and ec's
# s = ['MLPPWTLGLLLLATVRGKEVCYGQLGCFSDEKPWAGTLQRPVKLLPWSPEDIDTRFLLYTNENPNNFQLITGTEPDTIEASNFQLDRKTRFIIHGFLDKAEDSWPSDMCKKMFEVEKVNCICVDWRHGSRAMYTQAVQNIRVVGAETAFLIQALSTQLGYSLEDVHVIGHSLGAHTAAEAGRRLGGRVGRITGLDPAGPCFQDEPEEVRLDPSDAVFVDVIHTDSSPIVPSLGFGMSQKVGHLDFFPNGGKEMPGCKKNVLSTITDIDGIWEGIGGFVSCNHLRSFEYYSSSVLNPDGFLGYPCASYDEFQESKCFPCPAEGCPKMGHYADQFKGKTSAVEQTFFLNTGESGNFTSWRYKISVTLSGKEKVNGYIRIALYGSNENSKQYEIFKGSLKPDASHTCAIDVDFNVGKIQKVKFLWNKRGINLSEPKLGASQITVQSGEDGTEYNFCSSDTVEENVLQSLYPC ',
#     'MIGRLNHVAIAVPDLEAAAAQYRNTLGAEVGAPQDEPDHGVTVIFITLPNTKIELLHPLGEGSPIAGFLEKNPAGGIHHICYEVEDILAARDRLKEAGARVLGSGEPKIGAHGKPVLFLHPKDFNGCLVELEQV']
# ec = ['4.1.1.15', '5.1.99.-']

sequence_Example = "A E T C Z A O"
sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
encoded_input = tokenizer(sequence_Example, return_tensors='tf')
print(encoded_input)
output = model(**encoded_input)
print(output)

# working prot_XLNet pretrained model, loading PyTorch chekpoint into tensorflow model.
