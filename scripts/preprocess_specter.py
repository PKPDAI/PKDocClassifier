""" This file gets all the abstracts and titles and puts them in the format specified at
https://github.com/allenai/specter """

import pandas as pd
import os
import json


def processthem(inp_path):
    data_df = pd.read_parquet(inp_path)
    entries = data_df.to_dict('records')
    meta_ready = {str(entry['pmid']): dict(title=entry['title'], abstract=entry['abstract'],
                                           paper_id=str(entry['pmid'])) for entry in entries}
    ids_ready = [entry['pmid'] for entry in entries]
    return ids_ready, meta_ready


def save_ids(inp_list, output_dir, out_file):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    out_name = os.path.join(output_dir, out_file)

    with open(out_name, 'w') as f:
        for temp_item in inp_list:
            f.write('%i\n' % temp_item)


def save_json(inp_dict, output_dir, out_file):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    out_name = os.path.join(output_dir, out_file)

    with open(out_name, 'w', encoding='utf-8') as f:
        json.dump(inp_dict, f, ensure_ascii=False)


if __name__ == '__main__':
    # 1. Paths
    out_dir = os.path.join("data", "encoded", "specter")
    path_dev = os.path.join("data", "subsets", "dev_subset.parquet")
    path_test = os.path.join("data", "subsets", "test_subset.parquet")
    # 2. Output paths
    dev_ids, dev_meta = processthem(path_dev)
    test_ids, test_meta = processthem(path_test)

    save_ids(inp_list=dev_ids, output_dir=out_dir, out_file="dev_ids.ids")
    save_ids(inp_list=test_ids, output_dir=out_dir, out_file="test_ids.ids")

    save_json(inp_dict=dev_meta, output_dir=out_dir, out_file="dev_meta.json")
    save_json(inp_dict=test_meta, output_dir=out_dir, out_file="test_meta.json")
