# PKDocClassifier
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgh95/PKDocClassifier/blob/master/LICENSE)

[**PKDocClassifier**](#pkdocclassifier) | [**Reproduce our results**](#reproduce-our-results) | [**Make new predictions**](#make-new-predictions) | [**Citing**](#citation)




This repository contains custom pipes and models to classify scientific publications from PubMed depending on whether they report new pharmacokinetic (PK) parameters from _in vivo_ studies. The final pipeline retrieved more than 120K PK publications and runs weekly updates. All the retrieved data has been accessible at https://app.pkpdai.com/

# Reproduce our results

## 1. Installing dependencies 

You will need and environment with **Python 3.7 or greater**. We strongly recommend that you use an isolated Python environment (such as virtualenv or conda) to install the packages related to this project. Our default option will be to create a virtual environment with conda:
    
1. If you don't have conda follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=conda#regular-installation)

2. Run 

    ````
   conda create -n PKDocClassifier python=3.7
    ````

3. Activate it through
    ````
   source activate PKDocClassifier
    ````

Then, clone and access this repository on your local machine through:

````
git clone https://github.com/fgh95/PKDocClassifier
cd PKDocClassifier
````
If you are on MacOSX run: 

````
brew install libomp
````

Install all dependencies by running: 

````
pip install .
````

## 2. Data download - Optional

If you would like to reproduce the steps taken for data retrieval and parsing you will need to download the whole MEDLINE dataset and store it into a spark dataframe. 
However, you can also skip this step and use the parsed data available at [data/subsets/](https://github.com/fgh95/PKDocClassifier/tree/master/data/subsets). Alternatively, follow the steps at [pubmed_parser wiki](https://github.com/titipata/pubmed_parser/wiki/Download-and-preprocess-MEDLINE-dataset) and place the resulting `medline_lastview.parquet` file at _data/medline_lastview.parquet_. Then, change the [spark config file](https://github.com/fgh95/PKDocClassifier/blob/master/sparksetup/sparkconf.py) to your spark configuration and run:

````
python getready.py
````

This should generate the files at [data/subsets/](https://github.com/fgh95/PKDocClassifier/tree/master/data/subsets).

## 3. Run

### 3.1. Field analysis and N-grams

3.1.1. To generate the features run (~30min):

````
python features_bow.py
````

3.1.2. Bootstrap field analysis (~3h on 12 threads, requires at least 16GB of RAM)

````
python bootstrap_bow.py \
    --input-dir data/encoded/fields \
    --output-dir data/results/fields \
    --output-dir-bootstrap data/results/fields/bootstrap \
    --path-labels data/labels/dev_data.csv
````

3.1.3. Bootstrap n-grams (~3h on 12 threads, requires at least 16GB of RAM)

````
python bootstrap_bow.py \
    --input-dir data/encoded/ngrams \
    --output-dir data/results/ngrams \
    --output-dir-bootstrap data/results/ngrams/bootstrap \
    --path-labels data/labels/dev_data.csv
````

3.1.4. Display results

````
python display_results.py \
    --input-dir  data/results/fields\
    --output-dir data/final/fields
````

````
python display_results.py \
    --input-dir  data/results/ngrams\
    --output-dir data/final/ngrams
````

### 3.2. Distributed representations


3.2.1 Encode using [SPECTER](https://github.com/allenai/specter)

To generate the features with specter you can preprocess the data running: 

````
python preprocess_specter.py
````

This will generate the following input data as .ids and .json files at `data/encoded/specter/`. Finally, 
to generate the input features you will need to clone the [SPECTER](https://github.com/allenai/specter) repo and follow the instructions on [how to use the pretrained model](https://github.com/allenai/specter#how-to-use-the-pretrained-model). 
After cloning and installing SPECTER dependencies we used the following command from the specter directory to encode the documents: 

 ````
python scripts/embed.py \
--ids ../data/encoded/specter/dev_ids.ids --metadata ../data/encoded/specter/dev_meta.json \
--model ./model.tar.gz \
--output-file ../data/encoded/specter/dev_specter.jsonl \
--vocab-dir data/vocab/ \
--batch-size 16 \
--cuda-device -1
 ````

 ````
python scripts/embed.py \
--ids ../data/encoded/specter/test_ids.ids --metadata ../data/encoded/specter/test_meta.json \
--model ./model.tar.gz \
--output-file ../data/encoded/specter/test_specter.jsonl \
--vocab-dir data/vocab/ \
--batch-size 16 \
--cuda-device -1
 ````

This should output two files in the data directory: 
`/data/encoded/specter/dev_specter.jsonl` and `data/encoded/specter/test_specter.jsonl`

3.2.2 Encode with BioBERT

3.2.3
Then to run the boostrap: 

python bootstrap_dist.py \
    --is-specter true \
    --input-dir data/encoded/specter \
    --output-dir data/results/distributional \
    --output-dir-bootstrap data/results/fields/bootstrap \
    --path-labels data/labels/dev_data.csv
````

# Make new predictions

# Citation