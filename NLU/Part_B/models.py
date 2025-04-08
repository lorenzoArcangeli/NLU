import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import torch.nn as nn
import torch
from transformers import pipeline
from transformers import BertConfig, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer
# Vanilla
class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, bidirectional=bidirectional_mode, dropout_mode=False, dropout=0.7):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional_mode, batch_first=True)    
        self.dropout_mode = dropout_mode
        if self.dropout_mode:
            self.dropout = nn.Dropout(dropout) # → (possibly after the LSTM output or before the linear layers) to prevent overfitting
        if bidirectional:
            hid_size=hid_size*2
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)


        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout
        if self.dropout_mode:
            utt_encoded = self.dropout(utt_encoded)

        # Get the last hidden state
        if self.utt_encoder.bidirectional:
            last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        else:
            last_hidden = last_hidden[-1,:,:]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent


class BertFineTunedModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, bidirectional=bidirectional_mode, dropout_mode=False, dropout=0.7):
        super(BertFineTunedModelIAS, self).__init__()

        self.bert_model=BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

    def forward(self, utterance, seq_lengths):        
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.bert_model(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Apply dropout
        utt_encoded = self.dropout(utt_encoded)

        # Get the last hidden state
        last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent



# BERT-based model for joint intent classification and slot filling
class BertModelIAS(nn.Module):
    def __init__(self, out_slot, out_int, bert_model_name="bert-base-uncased", dropout=0.5):
        """
        Initialize the BERT-based model for joint intent classification and slot filling
        
        Args:
            out_slot: number of slots (output size for slot filling)
            out_int: number of intents (output size for intent classification)
            bert_model_name: name of the pre-trained BERT model
            dropout: dropout probability
        """
        super(BertModelIAS, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.slot_out = nn.Linear(self.hidden_size, out_slot)
        self.intent_out = nn.Linear(self.hidden_size, out_int)
        
        # Tokenizer for handling sub-tokenization
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, subword_to_word_ids=None):
        """
        Forward pass of the model
        
        Args:
            input_ids: input token ids (batch_size, seq_len)
            attention_mask: attention mask (batch_size, seq_len)
            token_type_ids: token type ids (batch_size, seq_len)
            subword_to_word_ids: mapping from subword tokens to word tokens (batch_size, seq_len)
                                 used to handle BERT's subword tokenization
        
        Returns:
            slots: slot logits (batch_size, classes, seq_len)
            intent: intent logits (batch_size, classes)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the sequence output and pooled output
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Handle subword tokenization for slot filling
        if subword_to_word_ids is not None:
            # Aggregate subword embeddings to get word-level embeddings
            batch_size, seq_len, hidden_size = sequence_output.shape
            word_level_output = torch.zeros_like(sequence_output)
            
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    word_idx = subword_to_word_ids[batch_idx, seq_idx].item()
                    if word_idx != -1:  # Skip special tokens like [CLS], [SEP]
                        word_level_output[batch_idx, word_idx] += sequence_output[batch_idx, seq_idx]
            
            # Use word-level embeddings for slot filling
            slots = self.slot_out(word_level_output)
        else:
            # If no subword mapping is provided, use the sequence output directly
            slots = self.slot_out(sequence_output)
        
        # Use pooled output for intent classification
        intent = self.intent_out(pooled_output)
        
        # Permute slots for loss computation
        slots = slots.permute(0, 2, 1)  # (batch_size, classes, seq_len)
        
        return slots, intent
    
    def tokenize_and_align(self, texts, word_labels=None):
        """
        Tokenize texts and align word labels with subword tokens
        
        Args:
            texts: list of input texts
            word_labels: list of word-level labels (optional)
            
        Returns:
            input_ids: token ids
            attention_mask: attention mask
            token_type_ids: token type ids
            subword_to_word_ids: mapping from subword tokens to word tokens
            word_ids_to_subword_ids: mapping from word tokens to subword tokens
        """
        batch_encoding = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = batch_encoding["input_ids"]
        attention_mask = batch_encoding["attention_mask"]
        token_type_ids = batch_encoding.get("token_type_ids", None)
        
        # Create mapping from subword tokens to word tokens
        subword_to_word_ids = []
        word_ids_to_subword_ids = []
        
        for i, text in enumerate(texts):
            # Get word_ids for each subword token
            word_ids = batch_encoding.word_ids(i)
            
            # Map from subword to word
            subword_to_word_map = []
            for word_id in word_ids:
                if word_id is None:
                    subword_to_word_map.append(-1)  # Special tokens like [CLS], [SEP]
                else:
                    subword_to_word_map.append(word_id)
            
            subword_to_word_ids.append(subword_to_word_map)
            
            # Map from word to first subword
            word_to_subword_map = {}
            for subword_idx, word_id in enumerate(word_ids):
                if word_id is not None and word_id not in word_to_subword_map:
                    word_to_subword_map[word_id] = subword_idx
            
            word_ids_to_subword_ids.append(word_to_subword_map)
        
        # Convert to tensors
        max_len = max(len(ids) for ids in subword_to_word_ids)
        padded_subword_to_word_ids = torch.ones((len(texts), max_len), dtype=torch.long) * -1
        
        for i, ids in enumerate(subword_to_word_ids):
            padded_subword_to_word_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        
        # Align word labels with subword tokens if provided
        aligned_labels = None
        if word_labels is not None:
            aligned_labels = []
            for i, (labels, word_to_subword_map) in enumerate(zip(word_labels, word_ids_to_subword_ids)):
                aligned = [-100] * len(batch_encoding["input_ids"][i])  # -100 is ignored in loss
                
                for word_idx, label in enumerate(labels):
                    if word_idx in word_to_subword_map:
                        subword_idx = word_to_subword_map[word_idx]
                        aligned[subword_idx] = label
                
                aligned_labels.append(aligned)
            
            aligned_labels = torch.tensor(aligned_labels, dtype=torch.long)
        
        return input_ids, attention_mask, token_type_ids, padded_subword_to_word_ids, aligned_labels