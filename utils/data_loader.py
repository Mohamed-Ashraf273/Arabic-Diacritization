import torch
from torch.utils.data import Dataset
import numpy as np

class DiacritizationDataset(Dataset):
    def __init__(self, X, y, letter2idx):
        self.X = X
        self.y = y
        self.pad_idx = letter2idx['<PAD>']
        self.space_idx = letter2idx[' ']
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_seq = np.asarray(self.X[idx])
        y_seq = np.asarray(self.y[idx])
        
        mask = np.zeros_like(x_seq, dtype=np.int64)
        
        pad_positions = np.where(x_seq == self.pad_idx)[0]
        if len(pad_positions) > 0:
            sentence_end = pad_positions[0]
        else:
            sentence_end = len(x_seq)

        space_positions = np.where(x_seq[:sentence_end] == self.space_idx)[0]
        all_boundaries = list(space_positions) + [sentence_end]
        
        start_idx = 0
        for boundary in all_boundaries:
            if start_idx < boundary:
                last_char_idx = boundary - 1
                if x_seq[last_char_idx] != self.pad_idx:
                    mask[last_char_idx] = 1
            start_idx = boundary + 1
        
        return (
            torch.tensor(x_seq, dtype=torch.long),
            torch.tensor(y_seq, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long)
        )