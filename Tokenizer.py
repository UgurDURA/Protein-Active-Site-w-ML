'''
Tokenizers for input and output. Save the configurations in input_tok_config.txt and output_tok_config.txt as json,
for the purpose of using them to detokenize later.

TODO:
> read from database to train tokenizers
> save tokenized data to new table in database?

'''
import tensorflow as tf
import json


# tokenizers
input_tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters=None, lower=False, char_level=True)
output_tok = tf.keras.preprocessing.text.Tokenizer(num_words=0, filters='#$%&()+/<=>?@\\^{|}~', lower=False, split='.', char_level=False)

# feed list of strings into fit_on_texts()
# s: example sequence
s = ['MLPPWTLGLLLLATVRGKEVCYGQLGCFSDEKPWAGTLQRPVKLLPWSPEDIDTRFLLYTNENPNNFQLITGTEPDTIEASNFQLDRKTRFIIHGFLDKAEDSWPSDMCKKMFEVEKVNCICVDWRHGSRAMYTQAVQNIRVVGAETAFLIQALSTQLGYSLEDVHVIGHSLGAHTAAEAGRRLGGRVGRITGLDPAGPCFQDEPEEVRLDPSDAVFVDVIHTDSSPIVPSLGFGMSQKVGHLDFFPNGGKEMPGCKKNVLSTITDIDGIWEGIGGFVSCNHLRSFEYYSSSVLNPDGFLGYPCASYDEFQESKCFPCPAEGCPKMGHYADQFKGKTSAVEQTFFLNTGESGNFTSWRYKISVTLSGKEKVNGYIRIALYGSNENSKQYEIFKGSLKPDASHTCAIDVDFNVGKIQKVKFLWNKRGINLSEPKLGASQITVQSGEDGTEYNFCSSDTVEENVLQSLYPC ',
'MIGRLNHVAIAVPDLEAAAAQYRNTLGAEVGAPQDEPDHGVTVIFITLPNTKIELLHPLGEGSPIAGFLEKNPAGGIHHICYEVEDILAARDRLKEAGARVLGSGEPKIGAHGKPVLFLHPKDFNGCLVELEQV']
ec = ['4.1.1.15', '5.1.99.-']

input_tok.fit_on_texts(s)
output_tok.fit_on_texts(ec)

input_config = input_tok.to_json()
output_config = output_tok.to_json()

with open('input_tok_config.txt', 'w') as outfile:
    json.dump(input_config, outfile)

with open('output_tok_config.txt', 'w') as outfile:
    json.dump(output_config, outfile)

