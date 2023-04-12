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
from accuracy import confusionMat 

parser = argparse.ArgumentParser(description='run SVM script')
parser.add_argument('-i', '--input', help='input folder path for data', required=True)
parser.add_argument('-c', '--crossfolds', help='number of crossfolds', required=False, default=5)
parser.add_argument('-j', '--json', help='json file with parameters for grid search cv', required=True)
parser.add_argument('-o', '--output', help='output pickle file path', required=True)
parser.add_argument('-r', '--results', help='path to results csv', required=True)
parser.add_argument('-n', '--numProcessors', help='number of processers', required=False, default=None)
parser.add_argument('-m', '--matrix', help='confusion matrix path', required=True)

if __name__ == '__main__':

    args = parser.parse_args()
    
    dataPath = args.input
    x_train = pd.read_csv(os.path.join(dataPath, './x_train.csv'))
    y_train = pd.read_csv(os.path.join(dataPath, './y_train.csv'))
    x_test = pd.read_csv(os.path.join(dataPath, './x_test.csv'))
    y_test = pd.read_csv(os.path.join(dataPath, './y_test.csv'))
    
    with open(args.json, 'r') as paramFile:
        dictionary = json.load(paramFile)
    exponents = dictionary['exponents']
    gamma = [10**n for n in exponents]
    C = [10**n for n in exponents]
    kernel = dictionary['kernel']
    param_grid = {'C': C, 'gamma': gamma, 'kernel': kernel}

    g = GridSearchCV(svm.SVC(probability=True), param_grid, refit = True, verbose = 3, cv=int(args.crossfolds), n_jobs=int(args.numProcessors))

    g.fit(x_train, y_train.isFolded)

    SVMclassifier = g.best_estimator_

    store(SVMclassifier, args.output)
    
    confusionMat(SVMclassifier, x_test, y_test.isFolded, args.matrix)

    acc = accuracy(SVMclassifier, x_test, y_test.isFolded)

    storeIt('SVM', f'{g.best_params_}', acc, args.output, args.results)
