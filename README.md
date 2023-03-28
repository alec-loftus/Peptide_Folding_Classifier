# Peptide_Folding_Classifier
We seek a classifier that will predict if a protein sequence (+3D coordinates) will fold. In this repository we are testing a variety of non-deep machine learning models to evaluate the folding propensity of IDPs. Simulated IDP data was provided by Dr. Peter Kekenes-Huskey for use in training models. Scikit-learn was used to train the ML models on the datasets. The data.py script generated training and testing subsets of data for use.

The necessary packages are listed in the 'requirements.txt' file and can be installed together by calling that file with your installer tool. Otherwise each tool can also be installed individually/manually.

The raw datasets provided can be found in the 'data' folder. These datasets were the input files for generating the training and testing models in the 'data.py' file. Replication of the experiment should be done using the files produced, which are the .csv files in the 'output' folder.
