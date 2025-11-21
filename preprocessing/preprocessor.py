class Preprocessor:
    def __init__(self, text):
        self.text = text

    def apply(self):
        raise NotImplementedError("Subclasses should implement this method")
