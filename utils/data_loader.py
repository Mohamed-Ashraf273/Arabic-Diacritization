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
    

class BOWDiacritizationDataset(Dataset):
    def __init__(self, X, y, bow_vectorizer, letter2idx, window_size=3):
        self.X = X
        self.y = y
        self.bow_vectorizer = bow_vectorizer
        self.letter2idx = letter2idx
        self.idx2letter = {v: k for k, v in letter2idx.items()}
        self.window_size = window_size
        self.pad_idx = letter2idx['<PAD>']
        self.space_idx = letter2idx[' ']
        
    def __len__(self):
        return len(self.X)
    
    def _get_bow_features_for_position(self, char_indices, position):
        start = max(0, position - self.window_size)
        end = min(len(char_indices), position + self.window_size + 1)
        context_chars = [self.idx2letter.get(idx, '<UNK>') for idx in char_indices[start:end]]
        bow_vector = self.bow_vectorizer.transform([context_chars])[0]
        return bow_vector
    
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
        
        bow_features = []
        for i in range(len(x_seq)):
            if x_seq[i] == self.pad_idx:
                bow_features.append(np.zeros(self.bow_vectorizer.vectorizer.vocabulary_.__len__()))
            else:
                bow_vec = self._get_bow_features_for_position(x_seq, i)
                bow_features.append(bow_vec)
        
        bow_features = np.array(bow_features, dtype=np.float32)
        
        return (
            torch.tensor(bow_features, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long)
        )