
import torch
import torch.nn as nn

#PART B.1
class LM_LSTM_TYING_WEIGHTS(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_TYING_WEIGHTS, self).__init__()
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
            raise ValueError("Different dimensions")
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        return output
    
class VariationalDropoutLayer(nn.Module):
    def __init__(self, dropout_init):
        super(VariationalDropoutLayer, self).__init__()
        self.dropout = dropout_init

    def forward(self, x):
        if not self.training:
            return x
        # generete mask
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask / (1 - self.dropout)  # Scaling per inverted dropout
        return x * mask


class LM_LSTM_VARIATIONAL_DROPOUT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.7,
                 emb_dropout=0.5, n_layers=1):
        super(LM_LSTM_VARIATIONAL_DROPOUT, self).__init__()
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