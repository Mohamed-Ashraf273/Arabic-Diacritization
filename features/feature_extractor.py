class FeatureExtractor:
    def __init__(self, text):
        self.text = text

    def extract_features(self):
        raise NotImplementedError("Subclasses should implement this method")