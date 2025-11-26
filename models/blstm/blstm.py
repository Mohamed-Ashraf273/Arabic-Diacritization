import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_size=256):
        super(BLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=13)

        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.bilstm(x) 
        x = self.fc(x)
        return x