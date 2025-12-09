import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_size=256, pretrained_embeddings=None):
        super(BiLSTM, self).__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx=13)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=13)

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths=None):
        x_emb = self.embedding(x)
        
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
        
        out = self.fc(lstm_out)
        return out