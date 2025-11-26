import torch
from torch.utils.data import Dataset
import numpy as np

class DiacritizationDataset(Dataset):
    def __init__(self, X, y, idx2letter):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.idx2letter = idx2letter
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_seq = self.X[idx]
        
        mask = torch.zeros_like(x_seq, dtype=torch.long)
        
        x_np = x_seq.numpy()
        
        pad_positions = np.where(x_np == 13)[0]
        if len(pad_positions) > 0:
            sentence_end = pad_positions[0]
        else:
            sentence_end = len(x_np)
        
        space_positions = np.where(x_np[:sentence_end] == 37)[0]
        all_boundaries = list(space_positions) + [sentence_end]
        
        start_idx = 0
        for boundary in all_boundaries:
            if start_idx < boundary:
                last_char_idx = boundary - 1
                if x_np[last_char_idx] != 13:
                    mask[last_char_idx] = 1
            start_idx = boundary + 1
        
        return self.X[idx], self.y[idx], mask