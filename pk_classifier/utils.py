import collections
import re
import string
import ujson
from pathlib import Path
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import torch
from transformers import BertTokenizer, BertModel


def make_pipeline(field_list, ngram):
    """
    Generates a pipeline as a list of sklearn transformers
    :param field_list: list of fields to include
    :param ngram: whether to include n grams
    :return: list of transformers
    """
    transformers = []
    for field in field_list:
        if field in ['title', 'abstract']:
            own_transformer = (field, Pipeline([
                ('selector', TextSelector(field)),
                ('preprocessor', BoWPreproc(field=field, ngram=ngram, mask_drugs=True))
            ])
                               )
        else:
            own_transformer = (field, Pipeline([
                ('selector', TextSelector(field=field)),
                ('preprocessor', MinimalPreproc(field=field))
            ]))
        transformers.append(own_transformer)
    return transformers


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
        docs = nlp.pipe(all_str)
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


def flatten(lst):  # From https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    """Works really efficiently for nested lists of lists/tuples. Needs Python 3+"""
    for el in lst:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def cleanup(token):
    """
    This function is specially relevant for detecting numeric vs non numeric parts. For instance  4323L/min gets
    tokenizes as ['##','L/min'] but 43.23 into ['##']. The rule behind this function is: after removing punctuations,
    if it only has numeric values, return '##'. However, if it starts with a numeric value but has a character string
    later on, split the numeric and non-numeric. Finally, if it doesn't start with a numeric value don't do anything.

    Might return single token or list of tokens (when it starts with numeric followed by character. So output
     needs to be flattened.
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
        raise Exception


def generate_ngrams(unigram_list, n, separator=" "):
    try:
        assert isinstance(unigram_list, list)
    except Exception:
        print("This is not a list of unigrams:", unigram_list)
        raise Exception

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
        raise Exception


class Concatenizer(BaseEstimator, TransformerMixin):
    def __init__(self, unifier):
        self.unifier = unifier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        conc_list = [self.unifier.join(doc) for doc in X]  # unifies all fields
        return pd.DataFrame(conc_list, columns=["BoW_Ready"])


def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open('r', encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


# ========== BioBERT encoders ================ #


def encode_bert(inp_text, nlp_model, bert_model, bert_tokenizer, maxmin):
    """
    Gets some input text as an input, runs spacy model to split into sentences, applies BERT encoding to each
    sub-word representation, and then it performs pooling of the whole sequence
    :param inp_text: string with the input text
    :param nlp_model: scipaCy model
    :param bert_model: loaded BioBERT model from transformers
    :param bert_tokenizer: loaded bioBERT tokenizer from transformers
    :param maxmin: boolean value that determines whether the min and max are included in the output pooling or it only
    does mean pooling
    """
    sentences_list = split_sentences(inp_text=inp_text, nlp_model=nlp_model)
    all_sentences = []
    for sentence in sentences_list:
        tokens_text, tokens_embed = tokens_emb(sentence=sentence, inp_tokenizer=bert_tokenizer, inp_model=bert_model)
        all_sentences.append(sentence_emb(tokens_embed))

    sentences_emb = torch.cat(all_sentences, dim=0)  # concatenate all tokens across sentences

    mean_sentence_embedding = torch.mean(sentences_emb, dim=0)  # take the mean across tokens
    max_sentence_embedding = torch.max(sentences_emb, dim=0, keepdim=False)[0]  # take the max across tokens
    min_sentence_embedding = torch.min(sentences_emb, dim=0, keepdim=False)[0]  # take the min across tokens
    if maxmin:
        all_pooling = [mean_sentence_embedding, max_sentence_embedding, min_sentence_embedding]
    else:
        all_pooling = [mean_sentence_embedding]

    pooling_concat = torch.cat(all_pooling, dim=0).numpy()
    return pooling_concat


def tokens_emb(sentence, inp_tokenizer, inp_model):
    """
    Tokenizes the input sentence and embeds each token into their embedding representation
    :param sentence: string sentence
    :param inp_tokenizer: bert tokenizer (transformers)
    :param inp_model: bert model (transformers)
    :return tokens as strings and tokens in embedding format
    """
    assert isinstance(sentence, str)
    inp_raw_tokens = inp_tokenizer.tokenize("[CLS]" + sentence + "[SEP]")
    tokens_tensor = torch.tensor(inp_tokenizer.encode(sentence)).unsqueeze(0)
    segments_ids = [1] * tokens_tensor.shape[1]
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = inp_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    inp_token_embeddings = torch.stack(hidden_states, dim=0)
    inp_token_embeddings = torch.squeeze(inp_token_embeddings, dim=1)
    inp_token_embeddings = inp_token_embeddings.permute(1, 0, 2)
    return inp_raw_tokens, inp_token_embeddings


def sentence_emb(inp_token_embeddings, n=4):
    """
    Adds the last 4 layers of each token and stacks all the input token representations except the first and last ones.
    It considers that the first and last tokens are special BERT tokens (CLS and SEP) and it drops them within the
    function
    :param inp_token_embeddings: torch tensor with all the token embeddings in each layer
    :param n: number of layers to add up
    :return: torch representation of all tokens stacked
    """
    inp_token_embeddings = inp_token_embeddings[1:-1]
    token_vecs_sum = torch.stack(
        [torch.sum(token[-n:], dim=0) for token in inp_token_embeddings])  # sum across last 4 layers
    return token_vecs_sum


def split_sentences(inp_text, nlp_model):
    """
    Splits an input string into sentence determined by spacy model
    :param inp_text: string with input text
    :param nlp_model: sciSpacy model
    :return: list of sentences in string format
    """
    doc = nlp_model(inp_text)
    sentences_list = [sentence.text for sentence in doc.sents]
    return sentences_list


class ConcatenizerEmb(BaseEstimator, TransformerMixin):
    def __init__(self, unifier):
        self.unifier = unifier

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        abstracts = X[:, -2]
        titles = X[:, -1]
        rest = np.delete(X, [X.shape[1] - 2, X.shape[1] - 1], axis=1)
        conc_list = [self.unifier.join(doc) for doc in rest]  # unifies all fields
        conc_list_df = pd.DataFrame(conc_list, columns=["BoW_Ready"])
        abstracts_df = pd.DataFrame(abstracts, columns=["abstract"])
        titles_df = pd.DataFrame(titles, columns=["title"])
        new_df = pd.concat([conc_list_df, abstracts_df, titles_df], axis=1)
        return new_df


class Embedder(BaseEstimator, TransformerMixin):
    def __init__(self, fields, maxmin):
        self.fields = fields
        self.maxmin = maxmin

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        abstracts = X[self.fields[0]].tolist()
        abstracts_checked = [str_text if not str_text == '' else 'xxxxx' for str_text in abstracts]
        titles = X[self.fields[1]].tolist()
        titles_checked = [str_text if not str_text == '' else 'xxxxx' for str_text in titles]
        nlp = spacy.load("en_ner_bc5cdr_md", disable=["ner"])

        # ==============================================
        #               Load BioBERT from huggingface
        # ==============================================

        path_model = "monologg/biobert_v1.1_pubmed"
        tokenizer = BertTokenizer.from_pretrained(path_model)
        model = BertModel.from_pretrained(path_model, output_hidden_states=True)
        model.eval()
        abs_docs_cl = [encode_bert(inp_text=input_text, nlp_model=nlp, bert_model=model, bert_tokenizer=tokenizer,
                                   maxmin=self.maxmin) for input_text in abstracts_checked]
        titles_docs_cl = [encode_bert(inp_text=input_text, nlp_model=nlp, bert_model=model, bert_tokenizer=tokenizer,
                                      maxmin=self.maxmin) for input_text in titles_checked]

        abs_docs_clean_s = pd.Series(abs_docs_cl)
        titles_docs_clean_s = pd.Series(titles_docs_cl)
        embedded_combied_df = pd.concat([X[self.fields[2]], abs_docs_clean_s, titles_docs_clean_s],
                                        axis=1)
        embedded_combied_df.columns = ['BoW_Ready', 'abstracts_embedded', 'titles_embedded']

        return embedded_combied_df

