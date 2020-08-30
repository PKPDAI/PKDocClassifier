# PKDocClassifier

This repository provides the code for developing a machine learning pipeline that classifies scientific publications from PubMed depending on whether they report new pharmacokinetic (PK) parameters from _in vivo_ studies.

In the analyses section the different pipelines..

The final pipeline retrieved more than 120K PK publications and runs weekly updates. All the retrieved data has been arranged at https://stage-app.pkpdai.com/. 

# Reproducing our results

## 1. Data download

If you would like to reproduce the steps taken for data retrieval and parsing you will need to download the whole MEDLINE dataset and store it into a spark dataframe. 
However, you can also skip this step and use the parsed data available at [data/subsets/](https://github.com/fgh95/PKDocClassifier/tree/master/data/subsets). Alternatively, follow the steps at [pubmed_parser wiki](https://github.com/titipata/pubmed_parser/wiki/Download-and-preprocess-MEDLINE-dataset) and place the resulting `medline_lastview.parquet` file (which the whole PubMED database) inside the `data` directory. Then, change the file to your spark configuration and run:

````
python getready.py
````



## 2. Installing dependencies 


## 3. Run