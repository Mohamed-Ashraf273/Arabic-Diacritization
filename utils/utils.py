import pickle
import re
import string
from pathlib import Path

import pickle
import re

from utils.data_loader import DiacritizationDataset
from torch.utils.data import DataLoader
from bs4 import BeautifulSoup

def separate_diacritics(text, diacritic2idx):
    base_chars = []
    char_diacritics = []
    
    i = 0
    while i < len(text):
        char = text[i]
        
        diacritic = ''
        j = i + 1
        while j < len(text) and text[j] in diacritic2idx:
            diacritic += text[j]
            j += 1
        
        base_chars.append(char)
        char_diacritics.append(diacritic)
        i = j
    
    return base_chars, char_diacritics

def preprocess(text, diacritics_chars):
    controls_to_remove = {'\u200f', '\u200e', '\u200b', '\u200c', '\u200d', 
                         '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',  '–', '–', '‒', '‐', '…', '🤷', '🏻', '♀', '½', '😑', '¼', '⅔'}
    
    text = ''.join([char for char in text if char not in controls_to_remove])
    
    split_punct = {",", ".", "،", ":", "?", "؟", "؛", "«", "»", "،"}
    
    text = BeautifulSoup(text, "html.parser").get_text("")
    
    url_pattern = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
    text = re.sub(url_pattern, "", text)
    
    text = re.sub(r"[A-Za-z]", "", text)
    
    text = re.sub(r"[0-9٠-٩]+", "", text)
    
    text = re.sub(r"ـ+", "", text)
    
    text = re.sub(rf"\s+([{diacritics_chars}])", r"\1", text)
    text = re.sub(f"({diacritics_chars})\\1+", r"\1", text)
    
    unwanted_punct = "".join(set(string.punctuation) - set(split_punct))
    text = re.sub(f"[{re.escape(unwanted_punct)}]", " ", text)
    
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join([line.strip() for line in text.split("\n")])
    
    return text

def create_data_pipeline(corpus_path, letter2idx, diacritic2idx, collate_fn, train=True, 
                        batch_size=32, bow_vectorizer=None, window_size=3):
    
    assert corpus_path.endswith(".txt"), "Corpus file must be a .txt file"

    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = f.read()

    utils_dir = Path(__file__).parent
    diacritics_path = utils_dir / 'diacritics.pickle'
    
    with open(diacritics_path, 'rb') as f:
        diacritics_chars = pickle.load(f)

    cleaned_data = preprocess(data, diacritics_chars)
    split_punct = {",", ".", "،", ":", "?", "؟", "؛", "«", "»", "،", "\n"}
    sentences = re.split(f"[{re.escape(''.join(split_punct))}]", cleaned_data)
    sentences = list(filter(lambda s: s.strip(), sentences))

    X = []
    y = []
    X_chars = []

    for sentence in sentences:
        chars, diacritics = separate_diacritics(sentence.strip(), diacritic2idx)
        
        X.append([letter2idx[char] for char in chars])
        y.append([diacritic2idx[diacritic] for diacritic in diacritics])
        if bow_vectorizer is not None:
            X_chars.append(chars)

    if bow_vectorizer is not None:
        from utils.data_loader import BOWDiacritizationDataset
        
        if train and bow_vectorizer.vectorizer is None:
            bow_vectorizer.fit(X_chars)
        
        dataset = BOWDiacritizationDataset(
            X, y, bow_vectorizer, letter2idx, window_size=window_size
        )
    else:
        dataset = DiacritizationDataset(X, y, letter2idx=letter2idx)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate_fn
    )

    if bow_vectorizer is not None:
        return dataset, data_loader, bow_vectorizer
    else:
        return dataset, data_loader
