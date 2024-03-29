import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
from tqdm import tqdm
import argparse
from pk_classifier.bootstrap import simple_tokenizer, TextSelector, f1_eval, update, read_in_bow, make_ids_per_test, \
    split_train_val_test
from pk_classifier.stats import plot_df


def process_them(input_tuple, rounds, test_prop, out_path_results, out_path_figure, out_path_bootstrap, use_idf):
    all_features, all_labs = read_in_bow(input_tuple[0], input_tuple[1])

    ids_per_test = make_ids_per_test(inp_df=all_labs)
    all_metrics_test = []
    optimal_epochs = []
    median_optimal_epochs = []
    median_f1s = []
    a = 0
    for round_i in tqdm(np.arange(rounds)):
        rd_seed = 10042006 + round_i
        per = test_prop

        # =====================================================================================================
        #               Make splits: 60% train, 20% validation, 20% temp test
        # ======================================================================================================

        x_train, x_val, x_test, y_train, y_val, y_test, pmids_train, pmids_val, pmids_test = \
            split_train_val_test(features=all_features, labels=all_labs, test_size=per, seed=rd_seed)

        # =====================================================================================================
        #               Decide max number of iterations using early stopping criteria on the validation set
        # ======================================================================================================

        balancing_factor = y_train.value_counts()["Not Relevant"] / y_train.value_counts()["Relevant"]
        if round_i == 0:
            print("Training with--- ", y_train.value_counts()["Relevant"], " ---Relevant instances")
        encoder = CountVectorizer(tokenizer=simple_tokenizer, ngram_range=(1, 1), lowercase=False, preprocessor=None,
                                  min_df=2)
        normalizer = TfidfTransformer(norm="l1", use_idf=use_idf)

        decoder = xgb.XGBClassifier(random_state=rd_seed, n_jobs=-1, n_estimators=2000, objective='binary:logistic',
                                    max_depth=4, learning_rate=0.1, colsample_bytree=1.,
                                    scale_pos_weight=balancing_factor, nthread=-1)

        # Define encoding pipeline
        enc_pip = Pipeline([
            ('encoder', FeatureUnion(transformer_list=[
                ('bow', Pipeline([
                    ('selector', TextSelector('BoW_Ready', emb=False)),
                    ('vect', encoder),
                    ('norm', normalizer)
                ])
                 )
            ]))
        ])

        x_train_features = enc_pip.fit_transform(x_train)
        x_val_features = enc_pip.transform(x_val)
        if a == 0:
            print("Using: ", x_train_features.shape[1], "features")
            a = 1
        eval_set = [(x_train_features, y_train), (x_val_features, y_val)]

        decoder.fit(x_train_features, y_train, eval_set=eval_set, verbose=False,
                    early_stopping_rounds=200, eval_metric=f1_eval)

        optimal_epochs.append(decoder.best_ntree_limit)
        median_epochs = np.median(optimal_epochs)
        median_optimal_epochs.append(median_epochs)

        if round_i in np.arange(0, rounds, 20):
            print("Median number of epochs:", median_epochs)

        # =====================================================================================================
        #               Apply predictions to the temp test set
        # ======================================================================================================

        x_test_encoded = enc_pip.transform(x_test)
        pred_test = decoder.predict(x_test_encoded)
        test_results = pd.DataFrame(pred_test == y_test.values, columns=['Result'])
        test_results['Result'] = test_results['Result'].astype(int)
        test_results['pmid'] = pmids_test.values

        """Update for output"""
        for index, x in test_results.iterrows():
            ids_per_test = update(x, ids_per_test)

        precision_test, recall_test, f1_test, support_test = precision_recall_fscore_support(y_test, pred_test,
                                                                                             average='binary',
                                                                                             pos_label="Relevant")

        current_metrics_test = [precision_test, recall_test, f1_test]
        all_metrics_test.append(current_metrics_test)
        all_f1s = [x[2] for x in all_metrics_test]
        temp_median_f1 = np.median(all_f1s)
        median_f1s.append(temp_median_f1)
        if round_i in np.arange(0, rounds, 20):
            print("Median F1 on test set:", temp_median_f1)

    df_results_test = pd.DataFrame(all_metrics_test, columns=['Precision', 'Recall', 'F1-score'])
    print("------------ FINAL RESULTS ------------")
    print("Validation Median:", df_results_test.median())
    print("Validation std:", df_results_test.std())
    df_results_test.to_csv(out_path_results)
    plot_df(df_results_test, out_path=out_path_figure)

    ids_per_test['CorrectlyClassified'] = ids_per_test[['times_correct']].div(ids_per_test.times_test, axis=0) * 100
    ids_per_val = ids_per_test.sort_values(by=['CorrectlyClassified'], ascending=True)
    ids_per_val.to_csv(out_path_bootstrap)


def run(input_dir: str, output_dir: str, output_dir_bootstrap: str, path_labels: str, overwrite: bool, use_idf: bool):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(output_dir_bootstrap):
        os.makedirs(output_dir_bootstrap, exist_ok=True)

    inp_dev_files = [inp_file for inp_file in os.listdir(input_dir) if 'test' not in inp_file]

    for inp_file in inp_dev_files:
        inp_path = os.path.join(input_dir, inp_file)
        experiment_name = inp_file.replace("dev_", "").replace(".parquet", "").replace("training_", "")
        if use_idf and "optimal" in experiment_name:
            experiment_name += "_idf"
        print("================== ", experiment_name, "=============================")
        # Define output
        if "res_" + experiment_name + ".csv" not in os.listdir(output_dir) or overwrite:
            print(os.listdir(output_dir))
            print("res_" + experiment_name + ".csv")
            out_res = os.path.join(output_dir, "res_" + experiment_name + ".csv")
            out_fig = os.path.join(output_dir_bootstrap, "res_" + experiment_name + ".png")
            out_dev = os.path.join(output_dir_bootstrap, "bootstrap_" + experiment_name + ".csv")
            inp_tuple = (inp_path, path_labels)
            process_them(input_tuple=inp_tuple, rounds=200, test_prop=0.2, out_path_results=out_res,
                         out_path_figure=out_fig, out_path_bootstrap=out_dev, use_idf=use_idf)
        else:
            print("Ignoring ", experiment_name, " since there is already results files in output directory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, help="The directory with files containing the encoded "
                                                      "documents in parquet format. It will iterate over all "
                                                      "files in this directory")

    parser.add_argument("--output-dir", type=str, help="Output directory to save the results of each bootstrap "
                                                       "iteration in a csv file.")

    parser.add_argument("--output-dir-bootstrap", type=str, help="Output directory to save the boostrap "
                                                                 "results as the misclassification "
                                                                 "rates per document during bootstrap.")

    parser.add_argument("--path-labels", type=str, help="Path to the csv containing the labels of the training "
                                                        "(dev) set")

    parser.add_argument("--overwrite", type=bool, help="Whether to overwrite files if results already present in the "
                                                       "output directory.", default=False)

    parser.add_argument("--use-idf", type=bool, help="Whether to use idf scores", default=False)

    args = parser.parse_args()
    run(input_dir=args.input_dir, output_dir=args.output_dir, output_dir_bootstrap=args.output_dir_bootstrap,
        path_labels=args.path_labels, overwrite=args.overwrite, use_idf=args.use_idf)


if __name__ == '__main__':
    main()
