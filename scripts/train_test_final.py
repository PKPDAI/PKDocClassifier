"""Script where the final pipeline is trained from scratch (preprocessing, encoding, decoding) with optimal
hyperparameters and applied to the test set"""

import argparse
import os
from typing import Dict

import joblib
from sklearn.metrics import precision_recall_fscore_support

from pk_classifier.bootstrap import Tokenizer, TextSelector, str2bool
from pk_classifier.utils import read_crossval, ConcatenizerEmb, Embedder, EmbeddingsJoiner
from pk_classifier.utils import make_preprocessing_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
import time
import xgboost as xgb


def train_final_pipeline(train_data: pd.DataFrame, train_labels: pd.DataFrame, best_params_cv: Dict) -> Pipeline:
    assert train_data.pmid.to_list() == train_labels.pmid.to_list()
    # ============ 1. Preprocessing and BERT embeddings pipeline =================================== #
    preprocess_and_bert_pipe = Pipeline([
        ('tokens', FeatureUnion(transformer_list=make_preprocessing_pipeline(field_list=["title", "abstract",
                                                                                         "mesh_terms",
                                                                                         "publication_types"], ngram=1),
                                n_jobs=-1)),
        ('tokens_conc', ConcatenizerEmb(" ;; ")),
        ('embedder', Embedder(fields=['abstract', 'title', 'BoW_Ready'], maxmin=False)),
        ('embeddings_joiner', EmbeddingsJoiner(out_colname='embedding'))
    ])

    # ============ 2. BoW encoding and classification pipeline =================================== #
    rd_seed = 10042006
    train_labels = train_labels['label']
    balancing_factor = train_labels.value_counts()["Not Relevant"] / train_labels.value_counts()["Relevant"]
    encoder = CountVectorizer(tokenizer=Tokenizer, ngram_range=(1, 1), lowercase=False, preprocessor=None,
                              min_df=best_params_cv['encoder__bow__vect__min_df'])
    normalizer = TfidfTransformer(norm="l1", use_idf=False)
    decoder = xgb.XGBClassifier(random_state=rd_seed,
                                n_estimators=148,
                                objective='binary:logistic',
                                learning_rate=0.1,
                                colsample_bytree=best_params_cv['clf__colsample_bytree'],
                                max_depth=best_params_cv['clf__max_depth'],
                                scale_pos_weight=balancing_factor, nthread=-1, n_jobs=-1)

    enc_dec_pipe = Pipeline([
        ('encoder', FeatureUnion(transformer_list=[
            ('bow', Pipeline([
                ('selector', TextSelector('BoW_Ready', emb=False)),
                ('vect', encoder),
                ('norm', normalizer)
            ])
             ),
            ('abs', Pipeline([
                ('selector', TextSelector('embedding', emb=True))
            ]))
        ])),
        ('clf', decoder)
    ])

    final_pipe = Pipeline([
        ('preprocess_and_embedder', preprocess_and_bert_pipe),
        ('bow_encoder_decoder', enc_dec_pipe)
    ])

    t1 = time.time()
    final_pipe.fit(train_data, train_labels)
    t2 = time.time()
    print("Overall time was: {}s, "
          "which approximates {}s per instance".format(round(t2 - t1, 2), round((t2 - t1) / len(train_data), 2)))

    return final_pipe


def predict_on_test(test_data: pd.DataFrame, test_labels: pd.DataFrame, optimal_pipeline: Pipeline) -> pd.DataFrame:
    assert test_data.pmid.to_list() == test_labels.pmid.to_list()
    test_labels = test_labels['label']
    print("Predicting test instances, this might take a few minutes...")
    pred_test = optimal_pipeline.predict(test_data)

    test_results = pd.DataFrame(pred_test == test_labels.values, columns=['Result'])
    accuracy = sum(test_results['Result'].values) / len(test_results)
    test_results['pmid'] = test_data['pmid']
    test_results['Correct label'] = test_labels
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(test_labels, pred_test,
                                                                              average='binary',
                                                                              pos_label="Relevant")
    print("===== Final results on the test set ==== ")
    print("Precision: {}\nRecall: {}\nF1: {}\nAccuracy: {}".format(precision_test, recall_test, f1_test, accuracy))
    test_results.sort_values(by=['Result'])
    return test_results


def run(path_train: str, train_labels: str, path_test: str, test_labels: str, cv_dir: str, output_dir: str,
        train_pipeline: bool):
    train_data = pd.read_parquet(path_train).sort_values(by=['pmid']).reset_index(drop=True)
    test_data = pd.read_parquet(path_test).sort_values(by=['pmid']).reset_index(drop=True)
    train_labels = pd.read_csv(train_labels).sort_values(by=['pmid']).reset_index(drop=True)
    test_labels = pd.read_csv(test_labels).sort_values(by=['pmid']).reset_index(drop=True)
    cv_results, cv_pipe = read_crossval(cv_dir=cv_dir)
    if train_pipeline:
        pipeline_trained = train_final_pipeline(train_data=train_data, train_labels=train_labels,
                                                best_params_cv=cv_pipe.best_params_)
        joblib.dump(pipeline_trained, os.path.join(output_dir, "optimal_pipeline.pkl"))
    else:
        pipeline_trained = joblib.load(os.path.join(output_dir, "optimal_pipeline.pkl"))

    test_results = predict_on_test(test_data=test_data, test_labels=test_labels, optimal_pipeline=pipeline_trained)
    test_results.to_csv(os.path.join(output_dir, "test_results_per_instance.csv"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path-train",
                        type=str,
                        help="Parquet file with the raw training data.",
                        default="../data/subsets/dev_subset.parquet")

    parser.add_argument("--train-labels",
                        type=str,
                        help="Path to the CSV file with the labels of the training data.",
                        default="../data/labels/dev_data.csv")

    parser.add_argument("--path-test",
                        type=str,
                        help="Parquet file with the raw test data.",
                        default="../data/subsets/test_subset.parquet")

    parser.add_argument("--test-labels",
                        type=str,
                        help="Path to the CSV file with the labels of the test data.",
                        default="../data/labels/test_data.csv")

    parser.add_argument("--cv-dir",
                        type=str,
                        help="Output directory of the cross-validation analysis.",
                        default="../data/results/final-pipeline")

    parser.add_argument("--output-dir",
                        type=str,
                        help="Directory to save the results and output files.",
                        default="../data/results/final-pipeline")

    parser.add_argument("--train-pipeline", type=str2bool, nargs='?',
                        help="Determine whether the input to process is from SPECTER",
                        const=True,
                        default=True)

    args = parser.parse_args()
    run(path_train=args.path_train, train_labels=args.train_labels, path_test=args.path_test,
        test_labels=args.test_labels, cv_dir=args.cv_dir, output_dir=args.output_dir,
        train_pipeline=args.train_pipeline)


if __name__ == '__main__':
    main()
