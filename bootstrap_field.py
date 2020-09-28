import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, precision_recall_fscore_support, make_scorer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from encoders.bootstrap import Tokenizer, TextSelector


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred), pos_label=1.)
    return 'f1_err', err


def read_in(path_preproc, path_labels):
    """Gets features and labels paths, reads in and checks that PMIDs correspond between files and that there are
    only 2 labels. Returns data as 2 pandas dataframes"""
    features = pd.read_parquet(path_preproc).sort_values(by=['pmid']).reset_index(drop=True)
    labs = pd.read_csv(path_labels).sort_values(by=['pmid']).reset_index(drop=True)
    assert all(labs['pmid'] == features['pmid'])
    assert len(labs['label'].unique()) == 2
    return features, labs


def update(entry, main_data):
    result = entry['Result']
    pmid = entry['pmid']
    position_main_data = int(np.where(main_data['pmid'] == pmid)[0])
    main_data.at[position_main_data, 'times_correct'] = main_data.iloc[position_main_data]['times_correct'] + result
    main_data.at[position_main_data, 'times_test'] = main_data.iloc[position_main_data]['times_test'] + 1
    return main_data


def processthem(inp_tuple, rounds, test_prop, out_path_results, out_path_figure, out_path_bootstrap):
    all_features, all_labs = read_in(inp_tuple[0], inp_tuple[1])

    all_metrics_test = []
    ids_per_test = pd.DataFrame(all_labs['pmid'], columns=['pmid'])
    ids_per_test['Dataset'] = all_labs['Dataset']
    ids_per_test['Real label'] = all_labs.label
    ids_per_test['times_correct'] = 0
    ids_per_test['times_test'] = 0

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

        x_train, x_val, y_train, y_val, pmids_train, pmids_val = train_test_split(all_features,
                                                                                  all_labs['label'],
                                                                                  all_labs['pmid'],
                                                                                  test_size=per,
                                                                                  shuffle=True,
                                                                                  random_state=rd_seed,
                                                                                  stratify=all_labs['label'])
        new_per = len(y_val) / len(y_train)
        x_train, x_test, y_train, y_test, pmids_train, pmids_test = train_test_split(x_train,
                                                                                     y_train,
                                                                                     pmids_train,
                                                                                     test_size=new_per,
                                                                                     shuffle=True,
                                                                                     random_state=rd_seed,
                                                                                     stratify=y_train)

        # =====================================================================================================
        #               Decide max number of iterations using early stopping criteria on the validation set
        # ======================================================================================================

        balancing_factor = y_train.value_counts()["Not Relevant"] / y_train.value_counts()["Relevant"]
        if round_i == 0:
            print("Training with--- ", y_train.value_counts()["Relevant"], " ---Relevant instances")
        encoder = CountVectorizer(tokenizer=Tokenizer, ngram_range=(1, 1), lowercase=False, preprocessor=None,
                                  min_df=2)
        normalizer = TfidfTransformer(norm="l1", use_idf=False)
        decoder = xgb.XGBClassifier(random_state=rd_seed, n_jobs=-1, n_estimators=2000, objective='binary:logistic',
                                    max_depth=4, learning_rate=0.1, colsample_bytree=1.,
                                    scale_pos_weight=balancing_factor, nthread=-1)

        # Define encoding pipeline
        EncPip = Pipeline([
            ('encoder', FeatureUnion(transformer_list=[
                ('bow', Pipeline([
                    ('selector', TextSelector('BoW_Ready', emb=False)),
                    ('vect', encoder),
                    ('norm', normalizer)
                ])
                 )
            ]))
        ])

        x_train_features = EncPip.fit_transform(x_train)
        x_val_features = EncPip.transform(x_val)
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

        x_test_encoded = EncPip.transform(x_test)
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
    plotit(df_results_test, out_path=out_path_figure, field=None)

    ids_per_test['CorrectlyClassified'] = ids_per_test[['times_correct']].div(ids_per_test.times_test, axis=0) * 100
    ids_per_val = ids_per_test.sort_values(by=['CorrectlyClassified'], ascending=True)
    ids_per_val.to_csv(out_path_bootstrap)


def plotit(df_results, out_path=None, field=None):
    f1s = df_results['F1-score'].values
    plt.figure(figsize=(10, 10))
    plt.hist(f1s, bins='auto')  # arguments are passed to np.histogram
    plt.title("F1-distribution")
    plt.ylabel('Absolute frequency')
    plt.xlabel('F1-score')
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()
        plt.close()


if __name__ == '__main__':

    # 1. Define input directory and labels (could make it an argparse)
    inp_dir = os.path.join("data", "encoded", "fields")
    out_dir = os.path.join("data", "results", "fields")
    out_dir_bootstrap = os.path.join("data", "results", "fields", "bootstrap")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(out_dir_bootstrap):
        os.makedirs(out_dir_bootstrap, exist_ok=True)
    path_labels_dev = os.path.join("data", "labels", "dev_data.csv")

    # 2. Run bootstrap
    for inp_file in os.listdir(inp_dir):
        inp_path = os.path.join(inp_dir, inp_file)
        experiment_name = inp_file.replace("dev_", "").replace(".parquet", "")
        print("================== ", experiment_name, "=============================")
        # Define output
        out_res = os.path.join(out_dir, "res_" + experiment_name + ".csv")
        out_fig = os.path.join(out_dir_bootstrap, "res_" + experiment_name + ".png")
        out_dev = os.path.join(out_dir_bootstrap, "bootstrap_" + experiment_name + ".csv")
        inp_tuple = (inp_path, path_labels_dev)
        processthem(inp_tuple=inp_tuple, rounds=200, test_prop=0.2, out_path_results=out_res, out_path_figure=out_fig,
                    out_path_bootstrap=out_dev)
