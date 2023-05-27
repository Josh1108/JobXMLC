from jobxmlc.registry import ENCODER, DATA_FILTER, register

@register(_name="tf-idf", _type=DATA_FILTER)
class tfidf10:
    def __init__(self):
        pass
    def preprocessing_data(self):
        return NotImplementedError
    def tokenizer(self):
        return NotImplementedError
    def create_embeddings(self):
        return NotImplementedError
    def save_embeddings(self):
        return NotImplementedError