import torch
import torch.nn as nn

class BiLSTM_BOW(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        self.vocab_size = vocab_size

        self.bilstm = nn.LSTM(
            vocab_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, bow_features, lengths=None):
        if lengths is not None:
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            x_sorted = bow_features[sorted_idx]
            
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, 
                lengths_sorted.cpu(), 
                batch_first=True, 
                enforce_sorted=True
            )
            
            lstm_packed, _ = self.bilstm(x_packed)
            
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_packed, 
                batch_first=True,
                padding_value=0.0,
                total_length=bow_features.size(1)
            )
            
            _, unsorted_idx = sorted_idx.sort()
            lstm_out = lstm_out[unsorted_idx]
        else:
            lstm_out, _ = self.bilstm(bow_features)
        
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out