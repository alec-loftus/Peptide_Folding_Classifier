# Import required libraries
import argparse
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from modelObject import store, openIt
import pandas as pd
import os
from accuracy import accuracy
from storePerformance import storeIt
from accuracy import confusionMat
import numpy as np

# Create an argument parser to parse command line arguments 
parser = argparse.ArgumentParser(description='Remove insignificant predictors; will retrain models')
parser.add_argument('-i', '--input', help='input file path to model object', required=True)
parser.add_argument('-c', '--cutoff', help='cutoff for selecting model parameters', required=False, default=0.01)
parser.add_argument('-f', '--folder', help='folder with all split data; must have x_train.csv, x_test.csv, y_train.csv, y_test.csv', required=True)
parser.add_argument('-o', '--output', help='output file to save new model', required=True)
parser.add_argument('-r', '--results', help='results.csv path', required=True)
parser.add_argument('-m', '--matrix', help='path to confusion matrix storage', required=True)
parser.add_argument('-t', '--title', help='title to store performance by in csv', required=False, default='default')

# Define a function to retrain the model on the provided training data 
def retrain(model, x_train, y_train):
    
    model.fit(x_train, y_train)
    return model

# Define a function to calculate the feature importances using permutation and return a sorted DataFrame
def importances(model, x_test, y_test):
    
    parameterImportances = {
            "names": np.array(list(x_test.columns)),
            "importances": np.array(list(permutation_importance(model, x_test, y_test, scoring='f1').importances_mean))
            }
    df = pd.DataFrame.from_dict(parameterImportances)
    return df.sort_values('importances')

# Entry point of the script
if __name__ == '__main__':
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Load the model from the input file
    model = openIt(args.input)

    # Load the training and testing data
    x_train = pd.read_csv(os.path.join(args.folder, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(args.folder, 'y_train.csv'))
    x_test = pd.read_csv(os.path.join(args.folder, 'x_test.csv'))
    y_test = pd.read_csv(os.path.join(args.folder, 'y_test.csv'))

    # Scale the training and testing data using MinMaxScaler
    scaler = MinMaxScaler()

    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)

    # Loop until all features have importance scores above the cutoff value
    boolean = True
    while boolean:
        
        # Calculate the feature importances
        parameters = importances(model, x_test, y_test)

        # Print the feature importances
        print(parameters)

        # Check if the feature with the lowest importance score is below the cutoff value
        indices = np.array([im <= float(args.cutoff) for im in parameters["importances"]])
        
        # If the feature with the lowest importance score is below the cutoff value, remoce it from the training and testing data, retrain the model, and repeat
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
