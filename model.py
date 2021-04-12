import torch


class Word2Vec(object):
    def __init__(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        super().__init__()
        self.sentences = sentences
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def train(self):
        pass

    def eval(self):
        pass
