from results.stats import get_all_results, make_results_table2
import os
import pandas as pd


if __name__ == '__main__':
    inp_dir = os.path.join("data", "results", "fields")
    out_dir = os.path.join("data", "final", "fields")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    all_files = os.listdir(inp_dir)
    result_files = [x for x in all_files if "res_" in x and ".csv" in x]
    get_all_results(result_files, inp_dir, out_dir)

    result = result_files[0]
    sample_df = pd.read_csv(os.path.join(inp_dir, result))
    sample_name = result.split(".")[0]

    results_table = make_results_table2(sample_df, sample_name)
