import pickle
import re
import string
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
                         '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',  '–', '–', '‒', '‐'}
    
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
    text = re.sub(f"[{re.escape(unwanted_punct)}]", "", text)
    
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join([line.strip() for line in text.split("\n")])
    
    return text