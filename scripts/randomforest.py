from sklearn.ensemble import RandomForestClassifier
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

parser = argparse.ArgumentParser(description='Run Random Forest script')
parser.add_argument('-i', '--input', help='input folder path for data', required=True)
parser.add_argument('-c', '--crossfolds', help='number of crossfolds', required=False, default=5)
parser.add_argument('-j', '--json', help='json file with parameters for grid search cv', required=True)
parser.add_argument('-o', '--output', help='output pickle file path', required=True)
parser.add_argument('-r', '--results', help='path to results csv', required=True)
parser.add_argument('-n', '--numProcessors', help='number of processers', required=False, default=None)
parser.add_argument('-m', '--matrix', help='confusion matrix path', required=True)
parser.add_argument('-a', '--curve', help='path for roc curve', required=False, default='outputROC.png')

if __name__ == '__main__':
    args = parser.parse_args()
    dataPath = args.input
    x_train = pd.read_csv(os.path.join(dataPath, './x_train.csv'))
    y_train = pd.read_csv(os.path.join(dataPath, './y_train.csv'))
    x_test = pd.read_csv(os.path.join(dataPath, './x_test.csv'))
    y_test = pd.read_csv(os.path.join(dataPath, './y_test.csv'))
    
    ### I HAD TO ADD THIS LINE TO RUN ON MY VM!! IT IS NOT IN THE KNN, PLEASE TAKE A LOOK
    y_train = y_train['isFolded'].astype(int).values
    y_test = y_test['isFolded'].astype(int).values
    #### ABOVE IS ADDED CODE TO RUN 


    with open(args.json, 'r') as paramFile:
        param_grid = json.load(paramFile)

    g = GridSearchCV(RandomForestClassifier(random_state=168), param_grid, refit=True, verbose=3, cv=int(args.crossfolds), n_jobs=int(args.numProcessors))
    g.fit(x_train, y_train)
    
    
    rf_classifier = g.best_estimator_
    store(rf_classifier, args.output)
    
    
    # calculate AUC and highest threshold
    area, threshold = roc(rf_classifier, x_test, y_test, args.curve)

    
    # create a confusion matrix and store it
    confusionMat(rf_classifier, x_test, y_test, args.matrix, threshold)

    # calculate F1 score and accuracy
    f1, acc = accuracy(rf_classifier, x_test, y_test, threshold)

    # calculate and store variable importances
    variable_importances = importances(rf_classifier, x_test, y_test).to_dict()

    storeIt('Random Forest', f'{g.best_params_}', {'AUC': area, 'f1score': f1, 'regularAccuracy': acc}, args.output, args.results, variable_importances)
