from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
import argparse
import json
from modelObject import store
from storePerformance import storeIt
from accuracy import accuracy
from accuracy import confusionMat, roc
from backwardselection import importances

#argparser allows for command line inputs to specify parameters and data paths
parser = argparse.ArgumentParser(description='run SVM script')
#-i allows input command; grabs multiple files from this path, so only input a folder; is required to run a script
parser.add_argument('-i', '--input', help='input folder path for data', required=True)
#crossfolds is how many dataset groupings to make; defaults to 5
parser.add_argument('-c', '--crossfolds', help='number of crossfolds', required=False, default=5)
#json specifies the parameters for gridsearchCV; is required
parser.add_argument('-j', '--json', help='json file with parameters for grid search cv', required=True)
#output designates the path
parser.add_argument('-o', '--output', help='output pickle file path', required=True)
#results can be directed anywhere, but it is recommended to path to the results csv to keep a running log of all models and their accuracy
parser.add_argument('-r', '--results', help='path to results csv', required=True)
parser.add_argument('-n', '--numProcessors', help='number of processers', required=False, default=4)
parser.add_argument('-m', '--matrix', help='confusion matrix path', required=True)
parser.add_argument('-a', '--curve', help='path for roc curve', required=False, default='outputROC.png')

if __name__ == '__main__':

    args = parser.parse_args()
    
    #dataPath variable set to whatever folder was designated as input
    dataPath = args.input
    
    #grabs the training and test data from the designated folder

    #script will fail if the input folder does not contain files with these names:

    x_train = pd.read_csv(os.path.join(dataPath, './x_train.csv'))
    y_train = pd.read_csv(os.path.join(dataPath, './y_train.csv'))
    x_test = pd.read_csv(os.path.join(dataPath, './x_test.csv'))
    y_test = pd.read_csv(os.path.join(dataPath, './y_test.csv'))

    #parameter file is pulled from parameters folder for the current script (svm here)
    with open(args.json, 'r') as paramFile:
        dictionary = json.load(paramFile)
    exponents = dictionary['exponents']
    gamma = [10**n for n in exponents]
    C = [10**n for n in exponents]
    kernel = dictionary['kernel']
    param_grid = {'C': C, 'gamma': gamma, 'kernel': kernel}
    #gridsearchCV performed using the parameters, refitting, and the specified crossfolds and processors (run with default settings if not specified)
    g = GridSearchCV(svm.SVC(probability=True), param_grid, refit = True, scoring='f1', verbose = 3, cv=int(args.crossfolds), n_jobs=int(args.numProcessors))
    #fits the gridsearchCV using training data
    g.fit(x_train, y_train)
    
    #SVMclassifier is the best estimator argument from sklearn tools
    SVMclassifier = g.best_estimator_
    
    #stores the estimator details as output
    store(SVMclassifier, args.output)
    

    # calculate AUC and highest threshold
    area, threshold = roc(SVMclassifier, x_test, y_test, args.curve)
    #create a confusion matrix on the x_ and y_test data; store the matrix in the user designated path
    confusionMat(SVMclassifier, x_test, y_test, args.matrix, threshold)
    #run the accuracy.py script to check accuracy of model's folding prediction
    f1, acc = accuracy(SVMclassifier, x_test, y_test, threshold)
    # record importances
    varaibleImportances = importances(SVMclassifier, x_test, y_test).to_dict()
    #record the model name, best parameters used, and accuracy in the results.csv file
    storeIt('SVM', f'{g.best_params_}', {'AUC': area, 'f1score': f1, 'regularAccuracy': acc, 'crossfoldScore': g.best_score_}, args.output, args.results, varaibleImportances)
