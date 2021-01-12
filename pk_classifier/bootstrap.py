from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix


def Tokenizer(str_input):
    return str_input.split(" ;; ")


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, emb):
        self.field = field
        self.emb = emb

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.emb:
            col = X[self.field].fillna('')
            docs = col.values
            documents = [doc for doc in docs]
            documents_ready = csr_matrix(documents)
            return documents_ready
        else:
            return X[self.field].fillna('')
