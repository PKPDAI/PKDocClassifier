# PKDocClassifier
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgh95/PKDocClassifier/blob/master/LICENSE) [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://app.pkpdai.com/) ![version](https://img.shields.io/badge/version-0.1.0-blue) 


[**PKDocClassifier**](#pkdocclassifier) | [**Reproduce our results**](#reproduce-our-results) | [**Make new predictions**](#make-new-predictions) | [**Citing**](#citation)



This repository contains custom pipes and models to classify scientific publications from PubMed depending on whether they estimate pharmacokinetic (PK) parameters from _in vivo_ studies. The final pipeline retrieved more than 120K PK publications and runs weekly updates available at https://app.pkpdai.com/.

# Reproduce our results

## 1. Installing dependencies 

You will need an environment with **Python 3.7+**. We strongly recommend that you use an isolated Python environment (such as virtualenv or conda) to install the packages related to this project. Our default option will be to create a virtual environment with conda:
    
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
   python scripts/features_bow.py
   ````

2. Bootstrap field analysis (~3h on 12 threads, requires at least 16GB of RAM)

   ````
   python scripts/bootstrap_bow.py \
       --input-dir data/encoded/fields \
       --output-dir data/results/fields \
       --output-dir-bootstrap data/results/fields/bootstrap \
       --path-labels data/labels/dev_data.csv
   ````

3. Bootstrap n-grams (~3h on 12 threads, requires at least 16GB of RAM)

   ````
   python scripts/bootstrap_bow.py \
       --input-dir data/encoded/ngrams \
       --output-dir data/results/ngrams \
       --output-dir-bootstrap data/results/ngrams/bootstrap \
       --path-labels data/labels/dev_data.csv
   ````

4. Display results

   ````
   python scripts/display_results.py \
       --input-dir  data/results/fields\
       --output-dir data/final/fields
   ````

   ````
   python scripts/display_results.py \
       --input-dir  data/results/ngrams\
       --output-dir data/final/ngrams
   ````

### 3.2. Distributed representations


1. Encode using [SPECTER](https://github.com/allenai/specter). To generate the features with specter you can preprocess the data running: 

   ````
   python preprocess_specter.py
   ````

   This will generate the following input data as .ids and .json files at `data/encoded/specter/`. Finally, 
   to generate the input features you will need to clone the [SPECTER](https://github.com/allenai/specter) repo and follow the instructions on [how to use the pretrained model](https://github.com/allenai/specter#how-to-use-the-pretrained-model). 
   After cloning and installing SPECTER dependencies we ran the following command from the specter directory to encode the documents: 

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


2. Generate BioBERT representations:

   ````
   python scripts/features_dist.py
   ````

3. Run bootstrap iterations for distributed representations:
   ````
   python scripts/bootstrap_dist.py \
       --is-specter True \
       --use-bow False \
       --input-dir data/encoded/specter \
       --output-dir data/results/distributional \
       --output-dir-bootstrap data/results/distributional/bootstrap \
       --path-labels data/labels/dev_data.csv \
       --path-optimal-bow data/encoded/ngrams/dev_unigrams.parquet
   ````
   
   ````
   python scripts/bootstrap_dist.py \
       --is-specter False \
       --use-bow False \
       --input-dir data/encoded/biobert \
       --output-dir data/results/distributional \
       --output-dir-bootstrap data/results/distributional/bootstrap \
       --path-labels data/labels/dev_data.csv \
       --path-optimal-bow data/encoded/ngrams/dev_unigrams.parquet
   ````

   
4. Display results

   ````
   python scripts/display_results.py \
       --input-dir  data/results/distributional \
       --output-dir data/final/distributional \
       --convert-latex
   ````
   
   ````
   python scripts/display_results.py \
       --input-dir  data/results/distributional/bow_and_distributional \
       --output-dir data/final/distributional/bow_and_distributional \
       --convert-latex
   ````

   From these plots we can see that the best-performing architecture on the training data, on average, is the one using average embeddings from BioBERT and unigram features. 


### 3.3. Final pipeline

Run the cross-validation analyses: 

   ````
   python scripts/cross_validate.py \
       --training-embeddings  data/encoded/biobert/dev_biobert_avg.parquet \
       --training-optimal-bow  data/encoded/ngrams/dev_unigrams.parquet \
       --training-labels  data/labels/dev_data.csv\
       --output-dir  data/results/final-pipeline \
   ````

Train the final pipeline (preprocessing, encoding, decoding) from scratch with optimal hyperparameters and apply it to the test set:

   ````
   python scripts/train_test_final.py \
       --path-train  data/subsets/dev_subset.parquet \
       --train-labels  data/labels/dev_data.csv \
       --path-test  data/subsets/test_subset.parquet \
       --test-labels  data/labels/test_data.csv \
       --cv-dir  data/results/final-pipeline \
       --output-dir  data/results/final-pipeline \
       --train-pipeline  True 
   ````

# Make new predictions

# Citation