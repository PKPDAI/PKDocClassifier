from features_bow import run as run_bow
import os
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline, FeatureUnion
from encoders.utils import make_pipeline, Embedder, ConcatenizerEmb


def pre_process(inp_path, out_path, field_list, ngrams, maxmin):
    """
    Preprocess data before encoding
    :param inp_path: path to the input file
    :param out_path: output file path
    :param field_list: list of fields/sections to include
    :param ngrams: number of ngrams to consider
    :param maxmin: whether to include the max and min in the BioBERT pooling
    :return: writes the parquet file in the out_path
    """
    data_df = pd.read_parquet(inp_path)
    encoding_pipeline = Pipeline([
        ('tokens', FeatureUnion(transformer_list=make_pipeline(field_list, ngrams), n_jobs=-1)),
        ('tokens_conc', ConcatenizerEmb(" ;; ")),
        ('embedder', Embedder(fields=['abstract', 'title', 'BoW_Ready'], maxmin=maxmin))
    ])
    preprocessed_df = encoding_pipeline.fit_transform(data_df)
    preprocessed_df['pmid'] = data_df['pmid']
    preprocessed_df.to_parquet(out_path)


def run(list_dict, input_file, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for inp_dict in tqdm(list_dict):
        maxmin = inp_dict["maxmin"]
        if "dev_" + inp_dict["name"] + ".parquet" not in os.listdir(output_dir):
            path_out_dev = os.path.join(output_dir, "dev_" + inp_dict["name"] + ".parquet")
            pre_process(inp_path=input_file, out_path=path_out_dev, field_list=inp_dict["field"],
                        ngrams=inp_dict["ngram"], maxmin=maxmin)


if __name__ == '__main__':
    # 1. Input paths and output dirs specter
    path_dev = os.path.join("data", "subsets", "dev_subset.parquet")
    out_dir_specter = os.path.join("data", "encoded", "specter")

    to_process = [dict(name="specter", field=["mesh_terms", "publication_types"], ngram=1)]
    # TODO:careful    run_bow(to_process, path_dev, out_dir_specter)

    # 2. Input paths and output dirs BioBERT

    out_dir_biobert = os.path.join("data", "encoded", "biobert")
    to_process2 = [dict(name="biobert_avg", field=["mesh_terms", "publication_types"], ngram=1, maxmin=False),
                   dict(name="biobert_all", field=["mesh_terms", "publication_types"], ngram=1, maxmin=True)]

    run(to_process2, path_dev, out_dir_biobert)
