# Peptide_Folding_Classifier
We seek a classifier that will predict if a protein sequence (+3D coordinates) will fold. In this repository we are testing a variety of non-deep machine learning models to evaluate the folding propensity of IDPs. Simulated IDP data was provided by Dr. Peter Kekenes-Huskey for use in training models. Scikit-learn was used to train the ML models on the datasets. The data.py script generated training and testing subsets of data for use.

The necessary packages are listed in the 'requirements.txt' file and can be installed together by calling that file with your installer tool. Otherwise each tool can also be installed individually/manually.
1. First, either install locally or create a python virtual environment or conda environment
  a. To create a python virtual environemnt in linux/unix (nice way to manage project dependencies versions without mixing with other project dependencies versions)
    - ensure you have python3-venv, and if not install with the following command: **apt-get install python3-venv**
    - create a virtual environment: **python3 -m venv [name of environemnt]**
    - activate the viritual environment: **source venv/bin/activate**
2. Then, install packages with the follwing command: **pip3 install -r requirements.txt**

The raw datasets provided can be found in the 'data' folder. These datasets were the input files for generating the training and testing models in the 'data.py' file. Replication of the experiment should be done using the files produced, which are the .csv files in the 'output' folder.


**In order to run analysis, the following is an example of how to run a model and store its results:**

usage: svm.py [-h] -i INPUT [-c CROSSFOLDS] -j JSON -o OUTPUT -r RESULTS [-n NUMPROCESSORS] -m MATRIX

run SVM script

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input folder path for data
  -c CROSSFOLDS, --crossfolds CROSSFOLDS
                        number of crossfolds
  -j JSON, --json JSON  json file with parameters for grid search cv
  -o OUTPUT, --output OUTPUT
                        output pickle file path
  -r RESULTS, --results RESULTS
                        path to results csv
  -n NUMPROCESSORS, --numProcessors NUMPROCESSORS
                        number of processers
  -m MATRIX, --matrix MATRIX
                        confusion matrix path

results stored in results folder and model object in models folder
