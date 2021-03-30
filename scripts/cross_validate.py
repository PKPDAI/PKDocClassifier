import argparse
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, make_scorer
from pk_classifier.bootstrap import read_in_distributional, simple_tokenizer, TextSelector
import xgboost as xgb
import pandas as pd
import joblib


HYPERPARAMETER_GRID = {
    'encoder__bow__vect__min_df': 2 ** np.arange(1, 10, 1),
    'clf__max_depth': 2 ** np.arange(1, 7, 1),
    'clf__colsample_bytree': [1 / 3, 2 / 3, 3 / 3],
}


def cross_validate_pk(training_embeddings: str, training_optimal_bow: str, training_labels: str, output_dir: str):
    all_features, all_labs = read_in_distributional(path_preproc=training_embeddings, path_labels=training_labels,
                                                    is_specter=False, path_optimal_bow=training_optimal_bow)
    all_labs = all_labs['label']
    rd_seed = 10042006

    # =====================================================================================================
    #               Define pipeline
    # ======================================================================================================
    hyp_grid = HYPERPARAMETER_GRID
    balancing_factor = all_labs.value_counts()["Not Relevant"] / all_labs.value_counts()["Relevant"]
    encoder = CountVectorizer(tokenizer=simple_tokenizer, ngram_range=(1, 1), lowercase=False, preprocessor=None)
    normalizer = TfidfTransformer(norm="l1", use_idf=False)
    decoder = xgb.XGBClassifier(random_state=rd_seed,
                                n_estimators=148,  # Median value of the optimal n_estimators across 200 bootstrapping
                                # iterations when early stopping was applied
                                objective='binary:logistic',
                                learning_rate=0.1,
                                scale_pos_weight=balancing_factor, nthread=-1, n_jobs=-1, )

    EncDecPip = Pipeline([
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

    f1_scorer = make_scorer(f1_score, pos_label="Relevant")
    grid_obj = GridSearchCV(EncDecPip, hyp_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=51)
    # Train
    model_trained = grid_obj.fit(all_features, all_labs)
    best_param = model_trained.best_params_
    best_param['Mean F1-score'] = model_trained.best_score_
    best_param['Time'] = model_trained.refit_time_
    print("Optimal parameter combination ", best_param)
    results_df = pd.DataFrame.from_dict(best_param, orient='index')
    # save your model and results
    results_df.to_pickle(os.path.join(output_dir, "cv_results.pkl"))
    joblib.dump(model_trained, os.path.join(output_dir, "cv_optimal_pipeline.pkl"))


def run(training_embeddings: str, training_optimal_bow: str, training_labels: str, output_dir: str):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    cross_validate_pk(training_embeddings=training_embeddings, training_optimal_bow=training_optimal_bow,
                      training_labels=training_labels, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-embeddings",
                        type=str,
                        help="Parquet file with the embeddings of the training data.",
                        default="../data/encoded/biobert/training_biobert_avg.parquet")

    parser.add_argument("--training-optimal-bow",
                        type=str,
                        help="Path to the parquet file with the optimal BoW features of the training set.",
                        default="../data/encoded/ngrams/training_unigrams.parquet")

    parser.add_argument("--training-labels",
                        type=str,
                        help="Path to the CSV file containing the labels of the training data.",
                        default="../data/labels/training_labels.csv")

    parser.add_argument("--output-dir",
                        type=str,
                        help="Directory to save the CV results and output files.",
                        default="../data/results/final-pipeline")

    args = parser.parse_args()
    run(training_embeddings=args.training_embeddings, training_optimal_bow=args.training_optimal_bow,
        training_labels=args.training_labels, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
