# Peptide_Folding_Classifier

We have created here a classifier that will predict if a protein sequence will fold. In this repository we test a variety of non-deep machine learning models and a deep learning model to evaluate the folding propensity of intrinsically disordered proteins (IDP)s. ML Models included in this repository are support vector machines (SVM), decision trees (DT), random forests (RF), k-nearest neighbors (KNN), and a deep learning feedforward neural network. Simulated IDP data was provided by Dr. Peter Kekenes-Huskey for use in training models. Scikit-learn was used to train the ML models on the datasets. 

The data.py script generated training and testing subsets of data for use, which are stored in the 'splitData' folder. The datasets we input into data.py can be found in the 'rawData' folder, but data.py can be run on any other data you wish to use for testing these models.

# Installing Packages

The necessary packages are listed in the 'requirements.txt' file and can be installed together by calling that file with your installer tool. Otherwise each tool can also be installed individually/manually.
1. First, either install locally or create a python virtual environment or conda environment
  - To create a python virtual environemnt in linux/unix (nice way to manage project dependencies versions without mixing with other project dependencies versions)
    - ensure you have python3-venv, and if not install with the following command: **apt-get install python3-venv**
    - create a virtual environment: **python3 -m venv [name of environemnt]**
    - activate the viritual environment: **source venv/bin/activate**
2. Then, install packages with the following command: **pip3 install -r requirements.txt**

# Running the scripts

To generate splitData from a different dataset, the data.py script can be run to split and properly format datasets for model testing. Alternatively, the model scripts can be run on the provided sample datasets in 'splitData' by calling the folder as the input.

In order to run analysis, the following is an example of how to run a model and store its results, with optional flags and inputs bracketed:

```
nohup python3 scripts/svm.py [-h] -i INPUT [-c CROSSFOLDS] -j JSON -o OUTPUT -r RESULTS [-n NUMPROCESSORS] -m MATRIX &> logFiles/nohupSVM.out &
```

Example test run of a model script:

```
nohup python3 scripts/svm.py -i splitData/ -j params/svm_params.json -o models/svmTEST.pkl -r results/results.csv -m results/svmTESTCM.png &
```

  - options:
    - -i INPUT, --input INPUT
                          input folder path for data (this is the splitData folder)
    - -c CROSSFOLDS, --crossfolds CROSSFOLDS
                          number of crossfolds (crossfold validation number of random splits for training across entire dataset)
    - -j JSON, --json JSON  
                          json file with parameters for grid search cv (these are in the params folder and depend per model as hyperparameters differ; will test all comibinations)
    - -o OUTPUT, --output OUTPUT
                          output pickle file path (store model object for future use and analysis)
    - -r RESULTS, --results RESULTS
                          path to results csv (this will store f1 score and basic model information)
    - -n NUMPROCESSORS, --numProcessors NUMPROCESSORS
                          number of processers (this is how many processers to use; recommend more because takes long time)
    - -m MATRIX, --matrix MATRIX
                          confusion matrix path (stores confusion matrix for basic evaluation)

Results are stored in the results folder and model object in the models folder. The results.csv must already be created with the following headers exactly: 

The raw datasets provided can be found in the 'data' folder. These datasets were the input files for the 'data.py' script which generated the training and testing data for model prediction. Replication of the experiment should be done using the files produced, which are the .csv files in the 'splitData' folder.


# Parameters

As explained above in the options, each model takes an input of its respective .json paramaters file. Viewing the .json file in the 'params' folder will show the parameters we have set for our data. If you wish to change the parameters, you can do so by editing the model's .json file or by creating an entirely new one and calling that from the command line instead.
