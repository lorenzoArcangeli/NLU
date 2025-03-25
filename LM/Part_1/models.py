import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

#PART A.1
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    
#PART A.2
class LM_LSTM_DROP(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.6,
                 emb_dropout=0.6, n_layers=1):
        super(LM_LSTM_DROP, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Dropout layer after embedding
        self.emb_dropout = nn.Dropout(emb_dropout)
        # Pytorch's LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Dropout layer before the last linear layer
        self.dropout = nn.Dropout(out_dropout)
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        drop1 = self.emb_dropout(emb)
        lstm_out, _ = self.lstm(drop1)
        drop2 = self.dropout(lstm_out)
        output = self.output(drop2).permute(0,2,1)
        return output


#PART B.1
class LM_LSTM_TYING_WEIGHTS(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        if hidden_size == emb_size:
            self.embedding.weight=self.output.weight
        else:
            raise ValueError("Differetn dimensions")
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    

class VariationalDropoutLayer(nn.Module):
    def __init__(self, dropout_init=0.1):
        super(VariationalDropoutLayer, self).__init__()
        # Parametro apprendibile per il tasso di dropout (log_alpha)
        # log_alpha controlla il livello di rumore: più basso è, meno dropout c'è
        self.log_alpha = nn.Parameter(torch.tensor([dropout_init]).log())

    def forward(self, x):
        # Generiamo rumore Bernoulli per il dropout
        # Il tasso di dropout è derivato da log_alpha
        p = torch.sigmoid(-self.log_alpha)  # Convertiamo log_alpha in una probabilità
        mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
        
        # Applichiamo il dropout moltiplicativo
        x_dropped = x * mask / (1 - p)
        return x_dropped

class LM_LSTM_VARIATIONAL_DROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # Aggiungiamo Variational Dropout dopo l'embedding
        self.emb_var_dropout = VariationalDropoutLayer(dropout_init=emb_dropout)
        
        # LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        
        # Aggiungiamo Variational Dropout dopo l'LSTM
        self.out_var_dropout = VariationalDropoutLayer(dropout_init=out_dropout)
        
        self.pad_token = pad_index
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # Embedding
        emb = self.embedding(input_sequence)
        
        # Applichiamo Variational Dropout agli embedding
        emb = self.emb_var_dropout(emb)
        
        # LSTM
        lstm_out, _ = self.lstm(emb)
        
        # Applichiamo Variational Dropout alle uscite dell'LSTM
        lstm_out = self.out_var_dropout(lstm_out)
        
        # Output
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
