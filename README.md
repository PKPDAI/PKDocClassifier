# PKDocClassifier
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fgh95/PKDocClassifier/blob/master/LICENSE)

This repository provides the code for developing a machine learning pipeline that classifies scientific publications from PubMed depending on whether they report new pharmacokinetic (PK) parameters from _in vivo_ studies.

In the analyses section the different pipelines..

The final pipeline retrieved more than 120K PK publications and runs weekly updates. All the retrieved data has been accessible at https://stage-app.pkpdai.com/. 

# Reproducing our results

First, clone this repository to your local machine through:

````
git clone https://github.com/fgh95/PKDocClassifier
cd PKDocClassifier
````



## 1. Installing dependencies 


## 2. Data download

If you would like to reproduce the steps taken for data retrieval and parsing you will need to download the whole MEDLINE dataset and store it into a spark dataframe. 
However, you can also skip this step and use the parsed data available at [data/subsets/](https://github.com/fgh95/PKDocClassifier/tree/master/data/subsets). Alternatively, follow the steps at [pubmed_parser wiki](https://github.com/titipata/pubmed_parser/wiki/Download-and-preprocess-MEDLINE-dataset) and place the resulting `medline_lastview.parquet` file at _data/medline_lastview.parquet_. Then, change the [spark config file](https://github.com/fgh95/PKDocClassifier/blob/master/sparksetup/sparkconf.py) to your spark configuration and run:

````
python getready.py
````

This should generate the files at [data/subsets/](https://github.com/fgh95/PKDocClassifier/tree/master/data/subsets).

## 3. Run
