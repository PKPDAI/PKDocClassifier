""" This script is only used once to select the documents that have been labeled and the pk subset.
It applies the functions from clean_parquet to clean the files and get them ready"""

from sparksetup import spark
import os
import pandas as pd
from pyspark.sql.types import IntegerType


def store_features(inp_path, out_path, all_pk_data):
    with open(inp_path, 'r') as myfile:
        inp_labels = pd.read_csv(myfile)
        pmids_lab = inp_labels['pmid'].values

    lab_docs = all_pk_data.filter(all_pk_data['pmid'].isin(pmids_lab.tolist()))
    lab_docs.write.parquet(out_path, mode="overwrite")


if __name__ == '__main__':
    
    # 0. Define data directory:
    data_dir = os.path.join("data")
    # 1. Load medline
    path_all = os.path.join(data_dir, "medline_lastview.parquet")
    all_sets = spark.read.parquet(path_all)
    all_sets = all_sets.withColumn('pmid', all_sets['pmid'].cast(IntegerType()))
    # 2. Input paths:
    path_labels_dev = os.path.join(data_dir, "labels", "training_labels.csv")
    path_labels_test = os.path.join(data_dir, "labels", "test_labels.csv")
    # 3. Output paths
    path_out_lab_dev = os.path.join(data_dir, "subsets", "training_subset.parquet")
    path_out_lab_test = os.path.join(data_dir, "subsets", "test_subset.parquet")
    # 4. Process files
    store_features(path_labels_dev, path_out_lab_dev, all_sets)
    store_features(path_labels_test, path_out_lab_test, all_sets)


