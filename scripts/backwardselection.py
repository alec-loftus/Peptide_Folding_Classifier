import argparse
from sklearn.inspection import permutation_importance
from modelObject import store, openIt
import pandas as pd
import os
from accuracy import accuracy
from storePerformance import storeIt
from accuracy import confusionMat
import numpy as np

parser = argparse.ArgumentParser(description='Remove insignificant predictors; will retrain models')
parser.add_argument('-i', '--input', help='input file path to model object', required=True)
parser.add_argument('-c', '--cutoff', help='cutoff for selecting model parameters', required=False, default=0.01)
parser.add_argument('-f', '--folder', help='folder with all split data; must have x_train.csv, x_test.csv, y_train.csv, y_test.csv', required=True)
parser.add_argument('-o', '--output', help='output file to save new model', required=True)
parser.add_argument('-r', '--results', help='results.csv path', required=True)
parser.add_argument('-m', '--matrix', help='path to confusion matrix storage', required=True)
parser.add_argument('-t', '--title', help='title to store performance by in csv', required=False, default='default')

def retrain(model, x_train, y_train):
    
    model.fit(x_train, y_train)
    return model

def importances(model, x_test, y_test):
    
    parameterImportances = {
            "names": np.array(list(x_test.columns)),
            "importances": np.array(list(permutation_importance(model, x_test, y_test, scoring='f1').importances_mean))
            }
    df = pd.DataFrame.from_dict(parameterImportances)
    return df.sort_values('importances')

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    model = openIt(args.input)

    x_train = pd.read_csv(os.path.join(args.folder, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(args.folder, 'y_train.csv')).isFolded
    x_test = pd.read_csv(os.path.join(args.folder, 'x_test.csv'))
    y_test = pd.read_csv(os.path.join(args.folder, 'y_test.csv')).isFolded

    boolean = True
    while boolean:
        
        parameters = importances(model, x_test, y_test)

        print(parameters)

        indices = np.array([im <= float(args.cutoff) for im in parameters["importances"]])

        if indices[0] != True:
            boolean = False
            break
        
        else:
            header = parameters.iloc[0,0]
            x_train = x_train.drop(header, axis=1)
            x_test = x_test.drop(header, axis=1)
            model = retrain(model, x_train, y_train)

    confusionMat(model, x_test, y_test, args.matrix)
    store(model, args.output)
    storeIt(args.title, f'{x_test.columns}', accuracy(model, x_test, y_test), args.output, args.results)
