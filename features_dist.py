from features_bow import run
import os

if __name__ == '__main__':

    # 1. Input paths and output dirs
    path_dev = os.path.join("data", "subsets", "dev_subset.parquet")
    out_dir_specter = os.path.join("data", "encoded", "specter")

    to_process = [dict(name="specter", field=["mesh_terms", "publication_types"], ngram=1)]
    run(to_process, path_dev, out_dir_specter)
