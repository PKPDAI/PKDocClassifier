import spacy
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.stem import PorterStemmer
import collections
import re
import sys
import string


class BoWPreproc(BaseEstimator, TransformerMixin):
    def __init__(self, field, ngram, mask_drugs):
        make_ident = dict(abstract="", title="T", affiliation="AF", author="AUTH", chemical_list="CH", medline_ta="J",
                          keywords="KW", mesh_terms="MSH", publication_types="TY")
        self.identifier = make_ident[field]
        self.unifier = " ;; "
        self.field = field
        self.ngram = ngram
        self.mask_drugs = mask_drugs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Convert to list of strings:
        all_str = X[self.field].tolist()
        # 2. Load spacy NER model for drugs:
        nlp = spacy.load("en_ner_bc5cdr_md")
        # 3. Apply model to all strings:
        docs = list(nlp.pipe(all_str))
        # 4. Preprocess text:
        all_ready = [preprocess(spacy_obj=doc, ident=self.identifier, unifier=self.unifier, ngram=self.ngram,
                                masking=self.mask_drugs) for doc in docs]
        return pd.DataFrame(all_ready)


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        col = X[[self.field]].fillna('')
        return col


class MinimalPreproc(BaseEstimator, TransformerMixin):
    """ This class is only used for fields that are not abstract or title in which very little preprocessing is
    applied"""

    def __init__(self, field):
        make_ident = dict(abstract="", title="T", affiliation="AF", author="AUTH", chemical_list="CH", medline_ta="J",
                          keywords="KW", mesh_terms="MSH", publication_types="TY")
        self.identifier = make_ident[field]
        self.unifier = " ;; "
        self.field = field

    #   self.stopwords = stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.field == 'medline_ta' or self.field == 'country':
            return X + self.identifier
        else:
            aslist = X[self.field].tolist()
            all_ready = [simple_preproc(str_list=str_list, identifier=self.identifier, unifier=self.unifier) for
                         str_list in aslist]
            return pd.DataFrame(all_ready)


def preprocess(spacy_obj, ident, unifier, ngram, masking):
    """Preprocesses an input spacy object for BoW encoding:
    Arguments:
        spacy_obj: class obtained after passing it through a spacy language model: nlp()
        ident: identifier to append at the end of each token. Useful to distinguish fields. e.g. pharmacokinetics vs
            pharmacokineticsT (abstract vs title mention). It will be different for each field
        unifier: the output of this function is a token-separated string. unifier will determine how to separate tokens
        ngram: size of ngrams
        masking: whether to mask drugnames
    """

    ps = PorterStemmer()

    # 1. Remove stopwords and punctuation
    spacy_obj_cl = [tok for tok in spacy_obj if (not tok.is_stop) and (not tok.is_punct)]

    # 2. Mask and lowercase
    if masking:  # @TODO: it'd be nice to replace multi-word drugnames with single word (
        # it would slow performance)
        tokens = ["DRUGNAME" if x.ent_type_ == "CHEMICAL" else x.text.lower() for x in spacy_obj_cl]
    else:
        tokens = [tok.lower_ for tok in spacy_obj_cl]  # Lowercase

    # 3. Cleanup inside tokens
    if tokens:
        clean_tokens = list(flatten([cleanup(tok) for tok in tokens]))  # Cleans and removes stopwords
    else:
        clean_tokens = []

    if clean_tokens == "" or clean_tokens == []:
        return ""

    else:
        # 4. Stemming
        stemmed_tokens = [ps.stem(tok) + ident for tok in clean_tokens]
        # 5. Append n-grams
        if ngram == 2:
            if len(stemmed_tokens) >= 2:
                bigrams = generate_ngrams(stemmed_tokens, ngram)
                stemmed_tokens = stemmed_tokens + bigrams
        elif ngram == 3:
            if len(stemmed_tokens) >= 3:
                bigrams = generate_ngrams(stemmed_tokens, ngram - 1)
                trigrams = generate_ngrams(stemmed_tokens, ngram)
                stemmed_tokens = stemmed_tokens + bigrams + trigrams

        # 6. Unify tokens with identifier
        single_string = unifier.join(stemmed_tokens)
        return single_string


def flatten(l):  # From https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    """Works really efficiently for nested lists of lists/tuples. Needs Python 3+"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def cleanup(token):
    """
    This function is specially relevant for detecting numeric vs non numeric parts. For instance  4323L/min gets
    tokenizs as ['##','L/min'] but 43.23 into ['##']. The rule behind this function is: after removing punctuations,
    if it only has numeric values, return '##'. However, if it starts with a numeric value but has a character string
    later on, split the numeric and non-numeric. Finally, if it doesn't start with a numeric value don't do anything.

    Might return single token or list of tokens (when it starts with numeric followed by character. So output
     needs to be flattened.

    Might return single token or list of tokens (when it starts with numeric followed by character. So output
     needs to be flattened
     """

    table = str.maketrans('', '', string.punctuation)  # define characters to be removed inside the token
    try:
        token = token.replace('\n', '')
        if hasNumbers(token):
            cleaned_token = token.translate(table)  # remove specific punctuation types inside token
        else:
            cleaned_token = token

        if cleaned_token == '':
            return ''
        else:

            if cleaned_token.isdigit():  # Entire token being a digit
                return "##"
            elif cleaned_token[0].isdigit():  # Initial part of the token is digit (but not all)
                # Split numeric and non-numeric parts (useful when units are attached to the value e.g. 4323L/min)
                # https://stackoverflow.com/questions/12409894/fast-way-to-split-alpha-and-numeric-chars-in-a-python-string
                match = re.findall(r"[^\W\d_]+|\d+", cleaned_token)
                if isinstance(match, list):
                    combo_list = ['##' if x.isdigit() else x for x in match]
                    return combo_list
                else:
                    raise ValueError("Problem in matching", cleaned_token, "with numeric + character")
            else:
                return cleaned_token
    except Exception:
        print("Error in cleaning up this token:", token)
        sys.exit()


def generate_ngrams(unigram_list, n, separator=" "):
    try:
        assert isinstance(unigram_list, list)
    except Exception:
        print("This is not a list of unigrams:", unigram_list)
        sys.exit()

    if len(unigram_list) < n:
        return unigram_list
    else:
        ngrams = zip(*[unigram_list[i:] for i in range(n)])
        return [separator.join(ngram) for ngram in ngrams]


def hasNumbers(inputString):
    """Checks whether the string has any digit value"""
    return any(char.isdigit() for char in inputString)


def simple_preproc(str_list, identifier, unifier):
    """Simple preprocessing applied only for metadata"""
    try:
        tokens = [token.strip() + identifier if not token == '' else '' for token in str_list.split(';')]
        re_unified = unifier.join(tokens)
        return re_unified

    except Exception:
        print("Error Processing this field:", identifier)
        print("And this string:", str_list)
        sys.exit()


class Concatenizer(BaseEstimator, TransformerMixin):
    def __init__(self, unifier):
        self.unifier = unifier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        conc_list = [self.unifier.join(doc) for doc in X]  # unifies all fields
        return pd.DataFrame(conc_list, columns=["BoW_Ready"])
