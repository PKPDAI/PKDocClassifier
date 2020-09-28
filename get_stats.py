import os
import pandas as pd


def get_stats(inp_path, inp_labels_path):
    labels = pd.read_csv(inp_labels_path)
    print("Displaying stats for ", inp_path.split("/")[-1])
    data_df = pd.read_parquet(inp_path)[
        ['title', 'abstract', 'author', 'journal', 'keywords', 'chemical_list', 'mesh_terms', 'publication_types',
         'affiliation']]
    res =  data_df.apply(lambda x: ((x.str.len() != 0) & (x is not None)).sum(), axis=0) / len(data_df) * 100
    print("Size: ", len(data_df))
    print("Proportions: ")
    print(labels['label'].value_counts() * 100 / len(labels))
    print(res)


if __name__ == '__main__':
    # 0. Data path

    # 1. Input paths
    path_dev = os.path.join("data", "subsets", "dev_subset.parquet")
    path_test = os.path.join("data", "subsets", "test_subset.parquet")
    path_labels_dev = os.path.join("data", "labels", "dev_data.csv")
    path_labels_test = os.path.join("data", "labels", "test_data.csv")
    get_stats(inp_path=path_dev, inp_labels_path=path_labels_dev)
    get_stats(inp_path=path_test, inp_labels_path=path_labels_test)
