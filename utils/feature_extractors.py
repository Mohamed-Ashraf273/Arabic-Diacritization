from sklearn.feature_extraction.text import CountVectorizer

class BOWVectorizer:
    def __init__(self, binary=False):
        self.vectorizer = None
        self.binary = binary
        
    def fit(self, tokenized_sentences):
        texts = [' '.join(tokens) for tokens in tokenized_sentences]
        
        self.vectorizer = CountVectorizer(
            token_pattern=r'\S+',
            binary=self.binary
        )
        self.vectorizer.fit(texts)
        
    def transform(self, tokenized_sentences):
        texts = [' '.join(tokens) for tokens in tokenized_sentences]
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, tokenized_sentences):
        self.fit(tokenized_sentences)
        return self.transform(tokenized_sentences)