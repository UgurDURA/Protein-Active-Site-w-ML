
#This file is created for experimentation

# To do list
'1- DATA PREPROCESSING '
' 1.1 Access the Data'
' 1.2 Spit the data into features and depandents'
' 1.3 At the end of the preprocessing the data split it into Test and Validation'
'2- TOKENIZATION'
' 2.1 BPE Algorithym'
' 2.2 '

from __future__ import print_function

import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import pandas as pd
import scipy as sp
# import sentencepiece as spm
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import sqlite3
from tensorflow.keras import layers

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_optimizer as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset


# Network definition
from model_def import ProteinClassifier
from data_prep import ProteinSequenceDataset
 
## SageMaker Distributed code.
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
import smdistributed.dataparallel.torch.distributed as dist

dist.init_process_group()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

con = sqlite3.connect('[DATA]/db/Enzymes.db')
cur = con.cursor()

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

cur.execute('SELECT sequence_string, ec_number_string FROM Entries')


Dataset= pd.read_csv('[DATA]/DB/Entries.csv')  #taking data from csv file, you can easily export the data from SQL file to csv
Ec_Number = list(Dataset.iloc[:,2].values) #features
Sequence = list(Dataset.iloc[:,-1].values)  #Dependent values      #a better name could be suggested

class ProteinSequenceDataset(Dataset):
    def __init__(self, sequence, targets, tokenizer, max_len):
        self.sequence = sequence
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'protein_sequence': sequence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

'Data Analysis'
 
class DataAnalysis(Sequence):
    def __init__(self,Sequence):
        count_aminos={}
        length_seqs=[]
        for i, seq in enumerate(Sequence):
            length_seqs.append(len(seq))
            for a in seq:
                if a in count_aminos:
                    count_aminos[a] += 1
                else:
                    count_aminos[a] = 0

        unique_aminos=list(count_aminos.keys())

        print('Unique aminos ({}):\n{}'.format(len(unique_aminos), unique_aminos))
        x=[i for i in range(len(unique_aminos))]
        plt.bar(x, count_aminos.values())
        plt.xticks(x, unique_aminos)
        print(list(count_aminos.values())[-5:])
        plt.show()

        print('Average length:', np.mean(length_seqs))
        print('Deviation:', np.std(length_seqs))
        print('Min length:', np.min(length_seqs))
        print('Max length:', np.max(length_seqs))

'Split dataset into test and validation'

def DataSplit(self,Sequence, Ec_number):

    #Important note for here, the split need to be performed after tokenization and embedding
    #I just added the code here 

        from sklearn.model_selection import train_test_split

        Ec_Number_train,Ec_Number_test,Sequence_train,Sequence_test =train_test_split(Ec_Number,Sequence,test_size=0.2, random_state=1)

        print(Ec_Number_train)
        print(Ec_Number_test)
        print(Sequence_train)
        print(Sequence_test)

            #Data splitted into test and training and saved into a text files as mentioned below

        SequenceTestFile = open("SequenceTest.txt", "w")
        for element in Sequence_test:
            SequenceTestFile.write(element + "\n")
            SequenceTestFile.close()

            print('Sequence Test Data Created Succesfully....')

            EcNumberTestFile = open("Ec_NumbersTest.txt", "w")
            for element in Ec_Number_test:
                EcNumberTestFile.write(element + "\n")
            EcNumberTestFile.close()

            print('Ec Number Test Data Created Succesfully....')

            SequenceTrainingFile = open("SequencesTraining.txt", "w")
            for element in Sequence_train:
                SequenceTrainingFile.write(element + "\n")
            SequenceTrainingFile.close()

            print('Sequence Training Data Created Succesfully....')

            EcNumberTrainingFile = open("Ec_NumbersTraining.txt", "w")
            for element in Ec_Number_train:
                EcNumberTrainingFile.write(element + "\n")
            EcNumberTrainingFile.close()

# print('Ec Number Training Data Created Succesfully....')
# def applyMask(dirtyData, dirty_idxs):
#     if (type(dirtyData)!=type(np.asarray([]))):
#         dirtyData=np.asarray(dirtyData)

#     returnData= dirtyData[np.logical_not(dirty_idxs)]

#     return returnData


"Model" 

PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
MAX_LEN = 512  # this is the max length of the sequence
PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)

class ProteinClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ProteinClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                        nn.Linear(self.bert.config.hidden_size, n_classes),
                                        nn.Tanh())
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        return self.classifier(output.pooler_output)


"Tokenization "

def Tokenizer(self):

    from model_def import ProteinClassifier




    MAX_LEN = 512  # this is the max length of the sequence
    PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)
        

    def model_fn(model_dir):

        logger.info('model_fn')
        print('Loading the trained model...')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ProteinClassifier(10) # pass number of classes, in our case its 10
        with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
            model.load_state_dict(torch.load(f, map_location=device))
        return model.to(device)

    def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        sequence = json.loads(request_body)
        print("Input protein sequence: ", sequence)
        encoded_sequence = tokenizer.encode_plus(
        sequence, 
        max_length = MAX_LEN, 
        add_special_tokens = True, 
        return_token_type_ids = False, 
        padding = 'max_length', 
        return_attention_mask = True, 
        return_tensors='pt'
        )
        input_ids = encoded_sequence['input_ids']
        attention_mask = encoded_sequence['attention_mask']

        return input_ids, attention_mask

    raise ValueError("Unsupported content type: {}".format(request_content_type))


    def predict_fn(input_data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        input_id, input_mask = input_data
        logger.info(input_id, input_mask)
        input_id = input_id.to(device)
        input_mask = input_mask.to(device)
        with torch.no_grad():
            output = model(input_id, input_mask)
            _, prediction = torch.max(output, dim=1)
            return prediction







# from transformers import BertForMaskedLM, BertTokenizer 
# import re

# model_name = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")

# tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
# # tokenizers = tf.saved_model.load(model_name)

# # tokens = tokenizers.en.lookup(encoded)
# # print(tokens)


# Result=[]

# for SequenceTokens in Sequence:
#     sequence_Example = re.sub(r"[UZOB]", "X", SequenceTokens)
#     encoded_input = tokenizer(sequence_Example, return_tensors='pt')
#     output = model_name(**encoded_input)
#     Result.append(output)


# TokenizedSequenceFile = open("TokenizedSequences.txt", "a")    
# for element in Result:

#     TokenizedSequenceFile.write(element + "\n")
#     TokenizedSequenceFile.close()
 

#Setup input pipeline

def _get_train_data_loader(batch_size, training_dir):
    dataset = pd.read_csv(os.path.join(training_dir, "[DATA]\DB\Entries.csv"))
    train_data = ProteinSequenceDataset(
        sequence=dataset.sequence.to_numpy(),
        targets=dataset.location.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank())
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                                  sampler=train_sampler)
    return train_dataloader

def _get_test_data_loader(batch_size, training_dir):
    dataset = pd.read_csv(os.path.join(training_dir, "deeploc_per_protein_test.csv"))
    test_data = ProteinSequenceDataset(
        sequence=dataset.sequence.to_numpy(),
        targets=dataset.location.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return test_dataloader

def freeze(model, frozen_layers):
    modules = [model.bert.encoder.layer[:frozen_layers]] 
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
            
def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = dist.get_local_rank()
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    if rank == 0:
        test_loader = _get_test_data_loader(args.test_batch_size, args.test)
        print("Max length of sequence: ", MAX_LEN)
        print("Freezing {} layers".format(args.frozen_layers))
        print("Model used: ", PRE_TRAINED_MODEL_NAME)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    model = ProteinClassifier(
        args.num_labels  # The number of output labels.
    )
    freeze(model, args.frozen_layers)
    model = DDP(model.to(device), broadcast_buffers=False)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    
    optimizer = optim.Lamb(
            model.parameters(), 
            lr = args.lr * dist.get_world_size(), 
            betas=(0.9, 0.999), 
            eps=args.epsilon, 
            weight_decay=args.weight_decay)
    
    total_steps = len(train_loader.dataset)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps)
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            outputs = model(b_input_ids,attention_mask=b_input_mask)
            loss = loss_fn(outputs, b_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            optimizer.zero_grad()
            
            if step % args.log_interval == 0 and rank == 0:
                logger.info(
                    "Collecting data from Master Node: \n Train Epoch: {} [{}/{} ({:.0f}%)] Training Loss: {:.6f}".format(
                        epoch,
                        step * len(batch['input_ids'])*world_size,
                        len(train_loader.dataset),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )
            if args.verbose:
                print('Batch', step, "from rank", rank)
        if rank == 0:
            test(model, test_loader, device)
        scheduler.step()
    if rank == 0:
        model_save = model.module if hasattr(model, "module") else model
        save_model(model_save, args.model_dir)

def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)
    logger.info(f"Saving model: {path} \n")

def test(model, test_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    loss_fn = nn.CrossEntropyLoss().to(device)
    tmp_eval_accuracy, eval_accuracy = 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            outputs = model(b_input_ids,attention_mask=b_input_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, b_labels)
            correct_predictions += torch.sum(preds == b_labels)
            losses.append(loss.item())
            
    print('\nTest set: Validation loss: {:.4f}, Validation Accuracy: {:.0f}%\n'.format(
        np.mean(losses),
        100. * correct_predictions.double() / len(test_loader.dataset)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--num_labels", type=int, default=10, metavar="N", help="input batch size for training (default: 10)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=1, metavar="N", help="input batch size for training (default: 1)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=8, metavar="N", help="input batch size for testing (default: 8)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=0.3e-5, metavar="LR", help="learning rate (default: 0.3e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, metavar="M", help="weight_decay (default: 0.01)")
    parser.add_argument("--seed", type=int, default=43, metavar="S", help="random seed (default: 43)")
    parser.add_argument("--epsilon", type=int, default=1e-8, metavar="EP", help="random seed (default: 1e-8)")
    parser.add_argument("--frozen_layers", type=int, default=10, metavar="NL", help="number of frozen layers(default: 10)")
    parser.add_argument('--verbose', action='store_true', default=False,help='For displaying SMDataParallel-specific logs')
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
   
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
