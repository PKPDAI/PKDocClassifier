import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_df(df_results, out_path=None):
    f1s = df_results['F1-score'].values
    plt.figure(figsize=(10, 10))
    plt.hist(f1s, bins='auto')
    plt.title("F1-distribution")
    plt.ylabel('Absolute frequency')
    plt.xlabel('F1-score')
    if out_path:
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()
        plt.close()


def make_results_table2(df_results, pipeline_name):
    df_results = df_results[["Precision", "Recall", "F1-score"]]
    percentiles = np.transpose(np.quantile(df_results, q=[0.025, 0.5, 0.975], axis=0))
    percentiles = np.round(percentiles, 4) * 100
    precision = "{0}({1},{2})".format(str(np.round(percentiles[0][1], 2)), str(np.round(percentiles[0][0], 2)),
                                      str(np.round(percentiles[0][2], 2)))
    recall = "{0}({1},{2})".format(str(np.round(percentiles[1][1], 2)), str(np.round(percentiles[1][0], 2)),
                                   str(np.round(percentiles[1][2], 2)))
    f1 = "{0}({1},{2})".format(str(np.round(percentiles[2][1], 2)), str(np.round(percentiles[2][0], 2)),
                               str(np.round(percentiles[2][2], 2)))

    iqv = np.round(percentiles[2][2], 2) - np.round(percentiles[2][0], 2)
    data = [[pipeline_name, precision, recall, f1, iqv]]
    out_dataframe = pd.DataFrame(data, columns=['Pipeline', 'Precision', 'Recall', 'F1-score', 'IQV'])
    return out_dataframe


def extract_f1(inp_text):
    return float(inp_text.split("(")[0])


def rename(inp_word):
    mapper = dict(res_title='Title', res_authors='Authors', res_journal='Journal', res_keywords='Keywords',
                  res_abstract='Title + Abstract', res_chemical='Chemicals', res_mesh='MeSH', res_pub_type='Pub. Type',
                  res_affiliations='Affiliations', res_all='All fields', res_optimal='Optimal Combination',
                  res_unigrams='Unigrams', res_bigrams='Bigrams', res_trigrams='Trigrams',
                  res_specter_alone='SPECTER', res_biobert_bow_mean='BioBERT mean pooling',
                  res_biobert_meanmaxmin='BioBERT mean \n + min + max pooling')
    if inp_word in mapper.keys():
        return mapper[inp_word]
    else:
        return inp_word


def get_all_results(inp_result_files, input_dir, output_dir):
    all_results = []
    all_for_boxplot = []
    all_for_boxplot_names = []
    for inp_result in inp_result_files:
        instance_df = pd.read_csv(os.path.join(input_dir, inp_result))
        all_for_boxplot.append(instance_df["F1-score"].values)
        instance_name = inp_result.split(".")[0]
        all_for_boxplot_names.append(instance_name)
        results = make_results_table2(instance_df, instance_name)
        all_results.append(results)

    all_results_ready = pd.concat(all_results)
    all_results_ready['F1'] = all_results_ready['F1-score'].apply(extract_f1)
    all_results_ready = all_results_ready.sort_values(by="F1-score", ascending=False)
    print(all_results_ready)
    all_results_ready.to_csv(os.path.join(output_dir, "all_results.csv"))

    idx_medians_sorted = np.asarray([np.median(x) for x in all_for_boxplot]).argsort()
    all_for_boxplot = [all_for_boxplot[i] * 100 for i in idx_medians_sorted]
    all_for_boxplot_names = [rename(all_for_boxplot_names[i]) for i in idx_medians_sorted]
    fig7, ax7 = plt.subplots()
    fig7.set_figheight(15)
    fig7.set_figwidth(15)
    ax7.boxplot(all_for_boxplot, labels=all_for_boxplot_names, whis=[2.5, 97.5])
    plt.ylabel('F-1 score (%)', fontsize=25)
    plt.xticks(rotation=75, fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
    plt.close()


