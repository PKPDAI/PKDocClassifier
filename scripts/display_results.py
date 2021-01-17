from pk_classifier.bootstrap import str2bool
from pk_classifier.stats import get_all_results
import os
import argparse


def run(inp_dir, out_dir, convert_latex):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    all_files = os.listdir(inp_dir)
    result_files = [x for x in all_files if "res_" in x and ".csv" in x]
    get_all_results(result_files, inp_dir, out_dir, convert_latex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, help="Directory with al the res_<>.csv files resulting from "
                                                            "bootstrap")
    parser.add_argument("-o", "--output-dir", type=str, help="Output directory for figures and tables")

    parser.add_argument("--convert-latex", type=str2bool, nargs='?',
                        help="Print the table in latex format",
                        const=True,
                        default=False)

    args = parser.parse_args()
    run(inp_dir=args.input_dir, out_dir=args.output_dir, convert_latex=args.convert_latex)


if __name__ == '__main__':
    main()
