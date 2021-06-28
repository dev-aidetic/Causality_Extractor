from gensim.models import KeyedVectors
import numpy as np
import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, "data")


class FastText:
    def __init__(self) -> None:
        model = KeyedVectors.load_word2vec_format(
            os.path.join(MODEL_PATH, "crawl-300d-2M.vec"), binary=False, encoding="utf8"
        )
        self.word_vectors = model.wv
        del model

    def get_sent_vectors(self, sent: str):
        tokens = sent.split()
        emb = []
        for token in tokens:
            emb.append(self.word_vectors[token])
        emb = np.mean(np.array(emb), axis=0)
        return emb
