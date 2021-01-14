import os
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from pk_classifier.utils import Embedder


def pre_process(inp_path, out_path, maxmin):
    """
    Preprocess data before encoding
    :param inp_path: path to the input file
    :param out_path: output file path
    :param maxmin: whether to include the max and min in the BioBERT pooling
    :return: writes the parquet file in the out_path
    """
    data_df = pd.read_parquet(inp_path)
    encoding_pipeline = Pipeline([
        #        ('tokens', FeatureUnion(transformer_list=make_pipeline(field_list, ngrams), n_jobs=-1)),
        #        ('tokens_conc', ConcatenizerEmb(" ;; ")),
        #       ('embedder', Embedder(fields=['abstract', 'title', 'BoW_Ready'], maxmin=maxmin))
        ('embedder', Embedder(fields=['abstract', 'title'], maxmin=maxmin))
    ])
    preprocessed_df = encoding_pipeline.fit_transform(data_df)
    preprocessed_df['pmid'] = data_df['pmid']
    preprocessed_df.to_parquet(out_path)


def mini_preprocess(inp_path, out_path):
    old_df = pd.read_parquet(inp_path)
    abstracts_mean = old_df.abstracts_embedded.apply(lambda x: x.tolist()[0:768])
    titles_mean = old_df.titles_embedded.apply(lambda x: x.tolist()[0:768])
    old_df['abstracts_embedded'] = abstracts_mean
    old_df['titles_embedded'] = titles_mean
    old_df.to_parquet(out_path)


def run(list_dict, input_file, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for inp_dict in tqdm(list_dict):
        maxmin = inp_dict["maxmin"]
        if "dev_" + inp_dict["name"] + ".parquet" not in os.listdir(output_dir):
            path_out_dev = os.path.join(output_dir, "dev_" + inp_dict["name"] + ".parquet")
            if inp_dict["name"] == "biobert_avg" and "dev_biobert_all.parquet" in os.listdir(output_dir):
                mini_preprocess(inp_path=os.path.join(output_dir, "dev_biobert_all.parquet"), out_path=path_out_dev)
            else:
                pre_process(inp_path=input_file, out_path=path_out_dev, maxmin=maxmin)
                #   field_list=inp_dict["field"],
                # ngrams=inp_dict["ngram"], maxmin=maxmin)


if __name__ == '__main__':
    path_dev = os.path.join("data", "subsets", "dev_subset.parquet")

    # 2. Process data for BioBERT

    out_dir_biobert = os.path.join("data", "encoded", "biobert")
    to_process2 = [dict(name="biobert_all", maxmin=True),
                   dict(name="biobert_avg", maxmin=False)
                   ]

    run(list_dict=to_process2, input_file=path_dev, output_dir=out_dir_biobert)
