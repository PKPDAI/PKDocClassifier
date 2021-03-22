import argparse
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
import numpy as np
from pk_classifier.utils import read_jsonl
import pandas as pd


def simple_tokenizer(str_input):
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


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred), pos_label=1.)
    return 'f1_err', err


def update(entry, main_data):
    result = entry['Result']
    pmid = entry['pmid']
    position_main_data = int(np.where(main_data['pmid'] == pmid)[0])
    main_data.at[position_main_data, 'times_correct'] = main_data.iloc[position_main_data]['times_correct'] + result
    main_data.at[position_main_data, 'times_test'] = main_data.iloc[position_main_data]['times_test'] + 1
    return main_data


def read_in_distributional(path_preproc, path_labels, is_specter, path_optimal_bow):
    """Gets features and labels paths, reads in and checks that PMIDs correspond between files and that there are
    only 2 labels. Returns data as 2 pandas dataframes"""
    if is_specter:
        emb_features = read_jsonl(path_preproc)
        emb_features = pd.DataFrame([(int(entry['paper_id']), entry['embedding']) for entry in emb_features],
                                    columns=['pmid', 'embedding']).sort_values(by=['pmid']).reset_index(drop=True)

    else:
        emb_features = pd.read_parquet(path_preproc).sort_values(by=['pmid']).reset_index(drop=True)
        emb_features['abstracts_embedded'] = emb_features.abstracts_embedded.apply(lambda x: x.tolist())
        emb_features['titles_embedded'] = emb_features.titles_embedded.apply(lambda x: x.tolist())
        emb_features['embedding'] = emb_features['abstracts_embedded'] + emb_features['titles_embedded']
        emb_features = emb_features.drop(['abstracts_embedded', 'titles_embedded'], axis=1)

    optimal_bow = pd.read_parquet(path_optimal_bow).sort_values(by=['pmid']).reset_index(drop=True)
    assert optimal_bow.pmid.to_list() == emb_features.pmid.to_list()
    emb_features['BoW_Ready'] = optimal_bow['BoW_Ready']
    features = emb_features
    labs = pd.read_csv(path_labels).sort_values(by=['pmid']).reset_index(drop=True)
    assert features['pmid'].to_list() == labs['pmid'].to_list()
    assert len(labs['label'].unique()) == 2
    return features, labs


def read_in_bow(path_preproc, path_labels):
    """Gets features and labels paths, reads in and checks that PMIDs correspond between files and that there are
    only 2 labels. Returns data as 2 pandas dataframes"""
    features = pd.read_parquet(path_preproc).sort_values(by=['pmid']).reset_index(drop=True)
    labs = pd.read_csv(path_labels).sort_values(by=['pmid']).reset_index(drop=True)
    features.pmid = features.pmid.astype('int64')
    labs.pmid = labs.pmid.astype('int64')
    assert features['pmid'].equals(labs['pmid'])
    assert len(labs['label'].unique()) == 2
    return features, labs


def str2bool(v):
    """
    Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_ids_per_test(inp_df: pd.DataFrame):
    ids_per_test = pd.DataFrame(inp_df['pmid'], columns=['pmid'])
    ids_per_test['Dataset'] = inp_df['Dataset']
    ids_per_test['Real label'] = inp_df.label
    ids_per_test['times_correct'] = 0
    ids_per_test['times_test'] = 0
    return ids_per_test


def split_train_val_test(features, labels, test_size, seed):
    """Splits a dataset into training, dev and test, and keeps dev and test the same size (test_size proportion).
    So if test_size is 0.2 (20%) it will split the input into 60% training, 20% dev and 20% test"""
    x_train, x_val, y_train, y_val, pmids_train, pmids_val = train_test_split(features,
                                                                              labels['label'],
                                                                              labels['pmid'],
                                                                              test_size=test_size,
                                                                              shuffle=True,
                                                                              random_state=seed,
                                                                              stratify=labels['label'])
    new_per = len(y_val) / len(y_train)
    x_train, x_test, y_train, y_test, pmids_train, pmids_test = train_test_split(x_train,
                                                                                 y_train,
                                                                                 pmids_train,
                                                                                 test_size=new_per,
                                                                                 shuffle=True,
                                                                                 random_state=seed,
                                                                                 stratify=y_train)
    return x_train, x_val, x_test, y_train, y_val, y_test, pmids_train, pmids_val, pmids_test


