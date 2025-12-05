import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=256, hidden_size=256, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=13)
        nn.init.xavier_uniform_(self.embedding.weight)
        
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x, lengths=None):
        x_emb = self.embedding(x)
        x_emb = self.emb_norm(x_emb)
        
        if lengths is not None:
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            x_sorted = x_emb[sorted_idx]
            
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
                total_length=x_emb.size(1)
            )
            
            _, unsorted_idx = sorted_idx.sort()
            lstm_out = lstm_out[unsorted_idx]
        else:
            lstm_out, _ = self.bilstm(x_emb)
        
        lstm_out = self.lstm_norm(lstm_out)
        out = self.fc(lstm_out)
        return out