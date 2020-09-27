""" This script trains & applies a pre-processing SKlern pipeline and saves the output for future analyses"""

import pandas as pd
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from encoders.utils import TextSelector, BoWPreproc, MinimalPreproc, Concatenizer
from tqdm import tqdm


def make_pipeline(field_list, ngram):
    transformers = []
    for field in field_list:
        if field in ['title', 'abstract']:
            own_transformer = (field, Pipeline([
                ('selector', TextSelector(field)),
                ('preprocessor', BoWPreproc(field=field, ngram=ngram, mask_drugs=True))
            ])
                               )
        else:
            own_transformer = (field, Pipeline([
                ('selector', TextSelector(field=field)),
                ('preprocessor', MinimalPreproc(field=field))
            ]))
        transformers.append(own_transformer)
    return transformers


def processthem(inp_path, out_path, field_list, ngrams):
    data_df = pd.read_parquet(inp_path)
    encoding_pipeline = Pipeline([
        ('tokens', FeatureUnion(transformer_list=make_pipeline(field_list, ngrams), n_jobs=-1)),
        ('tokens_conc', Concatenizer(" ;; "))
    ])
    preprocessed_df = encoding_pipeline.fit_transform(data_df)
    preprocessed_df['pmid'] = data_df['pmid']
    preprocessed_df.to_parquet(out_path)


if __name__ == '__main__':
    # 1. Input paths and output dir
    path_dev = os.path.join("data", "subsets", "dev_subset.parquet")
    out_dir = os.path.join("data", "encoded", "fields")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    # 2. Define Field combinations
    to_process = [dict(name="title", field=["title"], ngram=1),
                  dict(name="abstract", field=["title", "abstract"], ngram=1),
                  dict(name="authors", field=["title", "abstract", "author"], ngram=1),
                  dict(name="affiliations", field=["title", "abstract", "affiliation"], ngram=1),
                  dict(name="chemical", field=["title", "abstract", "chemical_list"], ngram=1),
                  dict(name="keywords", field=["title", "abstract", "keywords"], ngram=1),
                  dict(name="journal", field=["title", "abstract", "medline_ta"], ngram=1),
                  dict(name="mesh", field=["title", "abstract", "mesh_terms"], ngram=1),
                  dict(name="pub_type", field=["title", "abstract", "publication_types"], ngram=1)
                  ]

    for inp_dict in tqdm(to_process):
        path_out_dev = os.path.join(out_dir, "dev_" + inp_dict["name"] + ".parquet")
        processthem(inp_path=path_dev, out_path=path_out_dev, field_list=inp_dict["field"], ngrams=inp_dict["ngram"])
