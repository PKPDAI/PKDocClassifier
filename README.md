# PKDocClassifier
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgh95/PKDocClassifier/blob/master/LICENSE) ![version](https://img.shields.io/badge/version-0.0.1-blue) [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://app.pkpdai.com/)


[**PKDocClassifier**](#pkdocclassifier) | [**Reproduce our results**](#reproduce-our-results) | [**Make new predictions**](#make-new-predictions) | [**Citing**](#citation)



This repository contains custom pipes and models to classify scientific publications from PubMed depending on whether they report new pharmacokinetic (PK) parameters from _in vivo_ studies. The final pipeline retrieved more than 120K PK publications and runs weekly updates. All the retrieved data has been accessible at https://app.pkpdai.com/

# Reproduce our results

## 1. Installing dependencies 

You will need and environment with **Python 3.7 or greater**. We strongly recommend that you use an isolated Python environment (such as virtualenv or conda) to install the packages related to this project. Our default option will be to create a virtual environment with conda:
    
1. If you don't have anaconda installed follow the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=conda#regular-installation)

2. Create conda environment for this project and activate it 

    ````
   conda create -n PKDocClassifier python=3.7
   conda activate PKDocClassifier
    ````

3. Clone and access this repository on your local machine 
   
   ````
   git clone https://github.com/fgh95/PKDocClassifier
   cd PKDocClassifier
   ````
   **If you are on MacOSX install LLVM's OpenMP runtime library, e.g.** 

   ````
   brew install libomp
   ````

5. Install all project dependencies

   ````
   pip install .
   ````

## 2. Data download and parsing - Optional

If you would like to reproduce the steps taken for data retrieval and parsing you will need to download the whole MEDLINE dataset and store it into a spark dataframe. 
However, you can also skip this step and use the parsed data available at [data/subsets/](https://github.com/fgh95/PKDocClassifier/tree/master/data/subsets). Alternatively, follow the steps at [pubmed_parser wiki](https://github.com/titipata/pubmed_parser/wiki/Download-and-preprocess-MEDLINE-dataset) and place the resulting `medline_lastview.parquet` file at _data/medline_lastview.parquet_. Then, change the [spark config file](https://github.com/fgh95/PKDocClassifier/blob/master/sparksetup/sparkconf.py) to your spark configuration and run:

````
python getready.py
````

This should generate the files at [data/subsets/](https://github.com/fgh95/PKDocClassifier/tree/master/data/subsets).

## 3. Reproduce results

### 3.1. Field analysis and N-grams

1. To generate the features run (~30min):

   ````
   python features_bow.py
   ````

2. Bootstrap field analysis (~3h on 12 threads, requires at least 16GB of RAM)

   ````
   python bootstrap_bow.py \
       --input-dir data/encoded/fields \
       --output-dir data/results/fields \
       --output-dir-bootstrap data/results/fields/bootstrap \
       --path-labels data/labels/dev_data.csv
   ````

3. Bootstrap n-grams (~3h on 12 threads, requires at least 16GB of RAM)

   ````
   python bootstrap_bow.py \
       --input-dir data/encoded/ngrams \
       --output-dir data/results/ngrams \
       --output-dir-bootstrap data/results/ngrams/bootstrap \
       --path-labels data/labels/dev_data.csv
   ````

4. Display results

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

3.2.2 Run the following:

````
python features_dist.py
````

3.2.3
Then to run the boostrap for specter: 
````
python bootstrap_dist.py \
    --is-specter True \
    --input-dir data/encoded/specter \
    --output-dir data/results/distributional \
    --output-dir-bootstrap data/results/fields/bootstrap \
    --path-labels data/labels/dev_data.csv
````

# Make new predictions

# Citation