class Model:
    def __init__(self, name, version):
        self.name = name
        self.version = version

    def get_info(self):
        return f"Model Name: {self.name}, Version: {self.version}"
    
    def fit(self, data):
        raise NotImplementedError("Subclasses should implement this method")