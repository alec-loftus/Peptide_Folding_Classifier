from sklearn.neighbors import KNeighborsClassifier as knn
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

#imported argparse tool used to allow command line direction of inputs, paramaters, outputs for the knn script
parser = argparse.ArgumentParser(description='run KNN script')
#input flag to direct to the folder with the desired train/test data for use; required to run script
parser.add_argument('-i', '--input', help='input folder path for data', required=True)
#number of crossfolds to run knn; defaults to 5 if no specification is given
parser.add_argument('-c', '--crossfolds', help='number of crossfolds', required=False, default=5)
#json files located in 'params' folder can be specifically grabbed to perform gridsearchCV with desired specs
parser.add_argument('-j', '--json', help='json file with parameters for grid search cv', required=True)
#designates the output name and path
parser.add_argument('-o', '--output', help='output pickle file path', required=True)
#directs to a specific output file; it is recommended to direct to the results.csv file in the 'results' folder to keep all results together
parser.add_argument('-r', '--results', help='path to results csv', required=True)
#number of processors to use; defaults to none if no specification given
parser.add_argument('-n', '--numProcessors', help='number of processers', required=False, default=4)
#confusion matrix path direction; recommended in README.md to direct to 'results' folder
parser.add_argument('-m', '--matrix', help='confusion matrix path', required=True)

if __name__ == '__main__':
    #parser arguments assigned to args
    args = parser.parse_args()
    #input folder specified from argparse assigned to dataPath
    dataPath = args.input
    #x_ and y_ train and test data pulled from the input folder specified
    #knn script will fail if input folder has no data with these names
    x_train = pd.read_csv(os.path.join(dataPath, './x_train.csv'))
    y_train = pd.read_csv(os.path.join(dataPath, './y_train.csv'))
    x_test = pd.read_csv(os.path.join(dataPath, './x_test.csv'))
    y_test = pd.read_csv(os.path.join(dataPath, './y_test.csv'))

    #script will fail if the input folder does not contain files with these names:

    x_train = pd.read_csv(os.path.join(dataPath, './x_train.csv'))
    y_train = pd.read_csv(os.path.join(dataPath, './y_train.csv'))
    x_test = pd.read_csv(os.path.join(dataPath, './x_test.csv'))
    y_test = pd.read_csv(os.path.join(dataPath, './y_test.csv'))


    #user designated json paramater file is opened as paramFile
    with open(args.json, 'r') as paramFile:
        param_grid = json.load(paramFile)
    #gridsearchCV performed for knn using the parameter grid, refitting according to gridsearchCV from sklearn toolkit, and the designated or default crossfolds and numProcessors
    #assigned to g
    g = GridSearchCV(knn(), param_grid, refit = True, verbose = 3, cv=int(args.crossfolds), n_jobs=int(args.numProcessors))
    #fitting is done on the x and y training data
    g.fit(x_train, y_train)
    #best estimator argument is performed on the gridsearchCV and assigned to KNN classifier
    KNNclassifier = g.best_estimator_
    #store the KNNclasifier in the designated output folder
    store(KNNclassifier, args.output)
    # calculate AUC and highest threshold
    area, threshold = roc(model, x_test, y_test, rocfile)
    #create a confusion matrix on the x_ and y_test data; store the matrix in the user designated path
    confusionMat(KNNclassifier, x_test, y_test, args.matrix, threshold)
    #run the accuracy.py script to check accuracy of model's folding prediction
    acc = accuracy(KNNclassifier, x_test, y_test, threshold)
    #record the model name, best parameters used, and accuracy in the results.csv file
    storeIt('KNN', f'{g.best_params_}', {'AUC': area, 'f1score': acc}, args.output, args.results)
