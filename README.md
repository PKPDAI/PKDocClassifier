# PKDocClassifier
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgh95/PKDocClassifier/blob/master/LICENSE)

This repository contains custom pipes and models to classify scientific publications from PubMed depending on whether they report new pharmacokinetic (PK) parameters from _in vivo_ studies. The final pipeline retrieved more than 120K PK publications and runs weekly updates. All the retrieved data has been accessible at https://stage-app.pkpdai.com/. 

# Reproducing our results

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

1. To generate the different type of features

The first step will be to generate the input features for each experiment before running the bootstrapping analyses. 

