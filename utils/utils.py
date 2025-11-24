import pickle
import re
import string
from bs4 import BeautifulSoup

def remove_diacritics(text):
    with open("utils/diacritics.pickle", "rb") as f:
        diacritics_set = pickle.load(f)
    print("Loaded diacritics:", diacritics_set)
    diacritics_chars = "".join(list(diacritics_set))
    print(f"\nDiacritics as string: {diacritics_chars}")
    diacritics_pattern = f"[{re.escape(diacritics_chars)}]"
    return re.sub(diacritics_pattern, '', text)

def preprocess(text):
    split_punct = {",", ".", "،", ":", "?", "؟", "؛", "«", "»"}
    
    text = BeautifulSoup(text, "html.parser").get_text("")
    
    url_pattern = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
    text = re.sub(url_pattern, "", text)
    
    text = re.sub(r"[A-Za-z]", "", text)
    
    text = re.sub(r"[0-9٠-٩]+", "", text)
    
    text = re.sub(r"ـ+", "", text)
    
    with open("utils/diacritics.pickle", "rb") as f:
        diacritics_chars = "".join(list(pickle.load(f)))
        
    text = re.sub(rf"\s+([{diacritics_chars}])", r"\1", text)
    text = re.sub(f"({diacritics_chars})\\1+", r"\1", text)
    
    unwanted_punct = "".join(set(string.punctuation) - set(split_punct))
    text = re.sub(f"[{re.escape(unwanted_punct)}]", "", text)
    
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join([line.strip() for line in text.split("\n")])
    
    return text
