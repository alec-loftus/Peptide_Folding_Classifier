# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
import argparse
import json
# Import custom functions from other files
from modelObject import store
from storePerformance import storeIt
from accuracy import accuracy
from accuracy import confusionMat, roc
from backwardselection import importances

# Define command line arguments
parser = argparse.ArgumentParser(description='Run Decision Tree script')
parser.add_argument('-i', '--input', help='input folder path for data', required=True)
parser.add_argument('-c', '--crossfolds', help='number of crossfolds', required=False, default=5)
parser.add_argument('-j', '--json', help='json file with parameters for grid search cv', required=True)
parser.add_argument('-o', '--output', help='output pickle file path', required=True)
parser.add_argument('-r', '--results', help='path to results csv', required=True)
parser.add_argument('-n', '--numProcessors', help='number of processers', required=False, default=4)
parser.add_argument('-m', '--matrix', help='confusion matrix path', required=True)
parser.add_argument('-a', '--curve', help='path for roc curve', required=False, default='outputROC.png')

# Execute code if this script is the main script being run
if __name__ == '__main__':
    # Parse command line arguments
    args = parser.parse_args()
    # Read in data
    dataPath = args.input
    x_train = pd.read_csv(os.path.join(dataPath, './x_train.csv'))
    y_train = pd.read_csv(os.path.join(dataPath, './y_train.csv'))
    x_test = pd.read_csv(os.path.join(dataPath, './x_test.csv'))
    y_test = pd.read_csv(os.path.join(dataPath, './y_test.csv'))

    # Convert labels to intergers
    y_train = y_train['isFolded'].astype(int).values
    y_test = y_test['isFolded'].astype(int).values

    # Read in grid search parameters from json file
    with open(args.json, 'r') as paramFile:
        param_grid = json.load(paramFile)

    # Define and fit the decision tree model with grid search cross-validation
    g = GridSearchCV(DecisionTreeClassifier(random_state=123), param_grid, refit=True, verbose=3, cv=int(args.crossfolds), n_jobs=int(args.numProcessors))
    g.fit(x_train, y_train)
    # Save the best model
    dt_classifier = g.best_estimator_
    store(dt_classifier, args.output)
    
    # Calculate AUC and highest threshold
    area, threshold = roc(dt_classifier, x_test, y_test, args.curve)
    
    # Create confusion matrix and save it
    confusionMat(dt_classifier, x_test, y_test, args.matrix, threshold)
    # Calculate f1 score and accuracy and save them
    f1, acc = accuracy(dt_classifier, x_test, y_test, threshold)
    
    
    # Calculate and store variable importances
    variable_importances = importances(dt_classifier, x_test, y_test).to_dict()

    storeIt('Decision Tree', f'{g.best_params_}', {'AUC': area, 'f1score': f1, 'regularAccuracy': acc, 'crossfoldScore': g.best_score_}, args.output, args.results, variable_importances)
