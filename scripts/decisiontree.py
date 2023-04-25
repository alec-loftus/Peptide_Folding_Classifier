from sklearn.tree import DecisionTreeClassifier
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

parser = argparse.ArgumentParser(description='Run Decision Tree script')
parser.add_argument('-i', '--input', help='input folder path for data', required=True)
parser.add_argument('-c', '--crossfolds', help='number of crossfolds', required=False, default=5)
parser.add_argument('-j', '--json', help='json file with parameters for grid search cv', required=True)
parser.add_argument('-o', '--output', help='output pickle file path', required=True)
parser.add_argument('-r', '--results', help='path to results csv', required=True)
parser.add_argument('-n', '--numProcessors', help='number of processers', required=False, default=4)
parser.add_argument('-m', '--matrix', help='confusion matrix path', required=True)
parser.add_argument('-a', '--curve', help='path for roc curve', required=False, default='outputROC.png')


if __name__ == '__main__':
    args = parser.parse_args()
    dataPath = args.input
    x_train = pd.read_csv(os.path.join(dataPath, './x_train.csv'))
    y_train = pd.read_csv(os.path.join(dataPath, './y_train.csv'))
    x_test = pd.read_csv(os.path.join(dataPath, './x_test.csv'))
    y_test = pd.read_csv(os.path.join(dataPath, './y_test.csv'))


    y_train = y_train['isFolded'].astype(int).values
    y_test = y_test['isFolded'].astype(int).values

  
    with open(args.json, 'r') as paramFile:
        param_grid = json.load(paramFile)

    g = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, refit=True, verbose=3, cv=int(args.crossfolds), n_jobs=int(args.numProcessors))
    g.fit(x_train, y_train)

    dt_classifier = g.best_estimator_
    store(dt_classifier, args.output)
    
    # calculate AUC and highest threshold
    area, threshold = roc(dt_classifier, x_test, y_test, args.curve)
    
    
    confusionMat(dt_classifier, x_test, y_test, args.matrix, threshold)
    
    f1, acc = accuracy(dt_classifier, x_test, y_test, threshold)
    
    
    # calculate and store variable importances
    variable_importances = importances(dt_classifier, x_test, y_test).to_dict()

    storeIt('Decision Tree', f'{g.best_params_}', {'AUC': area, 'f1score': f1, 'regularAccuracy': acc}, args.output, args.results, variable_importances)
