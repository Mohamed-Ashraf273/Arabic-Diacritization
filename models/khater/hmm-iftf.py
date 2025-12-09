import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

class TfidfHMM:
    def __init__(self, n_components=3, n_iter=100):

        self.vectorizer = TfidfVectorizer(max_features=100)
        self.model = hmm.GaussianHMM(n_components=n_components, 
                                     covariance_type="diag",
                                     n_iter=n_iter)
        self.n_components = n_components
        
    def fit(self, sentences):
        # Apply TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(sentences)
        self.features = self.tfidf_matrix.toarray()
        
        # Train HMM on TF-IDF features
        self.model.fit(self.features)
        
        return self
    
    def predict_states(self, sentences):
        # Transform to TF-IDF
        features = self.vectorizer.transform(sentences).toarray()
        
        # Predict states
        states = self.model.predict(features)
        
        return states
    
    def score(self, sentences):
        features = self.vectorizer.transform(sentences).toarray()
        return self.model.score(features)
    
    def sample(self, n_samples=5):
        return self.model.sample(n_samples)

