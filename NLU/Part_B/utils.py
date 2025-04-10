import torch
import torch.utils.data as data
import json
from pprint import pprint
from collections import Counter
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "mps")
PAD_TOKEN = 0
CLS_TOKEN = 101
SEP_TOKEN = 102


def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    #added
    attention, _ = merge(new_item["attention"])
    token_type_id, _ = merge(new_item["token_type_id"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths

    #added
    new_item["attentions"] = attention
    new_item["token_type_ids"] = token_type_id
    return new_item


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        #self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        #self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    '''   
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    '''
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# First we get the 10% of the training set, then we compute the percentage of these examples 
def get_dev(tmp_train_raw, portion=0.10):
    portion = portion

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    
    return train_raw, dev_raw


'''
Transform each dataset sample to a dict with key:
-   utturance: textual phrase in ID sequence
-   slots: list of slots in ID sequence
-   intent: ID of intent class
'''

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, tokenizer, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = tokenizer
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        #self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        #self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

        self.utt_ids, self.slot_ids, self.attention_mask, self.token_id = self.mapping_seq(self.utterances, lang.word2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    #########
    # N.B → we use BERT to get word ID (token)
    #       we use LANG to obtain the ID about the slots and intents
    #
    # token_id = 0 if the token belong to the same sentence
    #            I can do this since in this case I'm not compering two sentences 
    #            Finally CLS and SEP are still present
    # subtoken → since a word can be splitted into more than one token
    #            I need to handle it. I assign the slop based on the first token
    #            and 'pad' to all the other subtokens in order to have the same length
    #########

    def mapping_seq(self, utterrance, slots, slot_mapper):
        res_utterrance = []
        res_slots = []
        res_attention_mask = []
        res_token_id = []

        for seq, slot in zip(utterrance, slots):
            tmp_utterance = []
            tmp_slots = []
            tmp_attention_mask = []
            tmp_token_type_id = []
            for word, element in zip(seq.split(), slot.split()):

                # tokenize word without special tokens
                #word_tokens = self.tokenizer(word, add_special_tokens=False)
                word_tokens = self.tokenizer(word)
                tmp_utterance.extend(word_tokens["input_ids"])
                
                #ad the id to the first token/word and the pad one for all the other tokens
                tmp_slots.append(slot_mapper[element])
                tmp_slots.extend(slot_mapper['pad']*len(word_tokens["input_ids"]-1))

                # attention mask and token type id 
                for i in range(len(word_tokens["input_ids"])):
                    tmp_attention_mask.append(1)
                    tmp_token_type_id.append(0)

        res_utterrance.append(tmp_utterance)
        res_slots.append(tmp_slots)
        res_attention_mask.append(tmp_attention_mask)
        res_token_id.append(tmp_token_type_id)

        return res_utterrance, res_slots, res_attention_mask, res_token_id
    '''
    # tokenization with Bert
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tokens = self.tokenizer.tokenize(seq)
            tmp_seq = [mapper[token] if token in mapper else mapper[self.unk] for token in tokens]
            res.append(tmp_seq)
        return res
    '''
