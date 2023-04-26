# import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.metrics import AUC
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from backwardselection import importances
import pandas as pd
import numpy as np
import argparse
from accuracy import confusionMat, roc, accuracy
from storePerformance import storeIt
from modelObject import store
import os
import json
import pdb

# creating command line parser for flags to run script from the command line in a customized manner
parser = argparse.ArgumentParser(description="train and evaluate a feed forward fully connected neural network")

# input folder path flag created
parser.add_argument('-i', '--input', help='input folder with data (should include scaled/normalized x_train, x_test, y_test, y_train CSVs', required=True)

# output file name flag created
parser.add_argument('-o', '--output', help='output file for model object; must be pkl', required=False, default='./output.pkl')

# results.csv path flag created
parser.add_argument('-r', '--results', help='path to results.csv (must have Name, Description, Metric, Path)', required=True)

# matrix path flag created
parser.add_argument('-m', '--matrix', help='path to matrix file; must be png', required=False, default='./outputCM.png')

# list of options of hyperparameters for tuning
parser.add_argument('-t', '--tuninghyperparameters', help='json file with lists of hyperparameters for each option (update with types that can be played around with)', required=False, default=None)

# options for roc file storage
parser.add_argument('-c', '--curve', help='file path to store ROC curve', required=False, default='./outputROC.png')

def createIt(inputSize=10, numHiddenLayers=2, numHiddenNodes='None', activationHidden='relu', optimizer='adam', loss='mse', rate=0.01):
    '''
    Creates model layers and assorts them together
    '''
    
    if numHiddenNodes == 'None':
        numHiddenNodes = inputSize+2

    i = Input(shape=(inputSize,))
    listLayers = []
    listLayers.append(i)
    for index in range(numHiddenLayers):
        l = Dense(numHiddenNodes, activation=activationHidden)(listLayers[-1])
        listLayers.append(l)
    o = Dense(1, activation='sigmoid')(listLayers[-1])
    
    model = Model(inputs=i, outputs=o)

    # returns overview of model shell
    model.summary()
    
    model.compile(optimizer=Adam(learning_rate=rate), loss=loss)

    return model

def train(model, x_train, y_train, x_test, y_test, epochs=1000, batch_size=20):
    '''
    Trains model on training data that is stored in the generator
    '''
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    

def hyperparameterTuning(function, x_train, y_train, param_grid, inputSize, cv=5, n_jobs=6, scoring='f1'):
    '''
    Selects best model from a park of models with a varied combindation of hyperparameters as specified by the user
    '''
    model = KerasClassifier(model=function, inputSize=inputSize, numHiddenLayers=2, numHiddenNodes='None', activationHidden='relu', optimizer='adam', loss='mse', rate=0.01)
    g = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs, cv=cv, scoring=scoring, refit=True, verbose=3)
    results = g.fit(x_train, y_train)

    return results.best_estimator_, results.best_params_

def evaluate(model, x_test, y_test, resultsFile, storepath, cmfile, rocfile, description="simple feed forward model"):
    '''
    Evaluates trained model on test data set and stores results
    '''

    area, threshold = roc(model, x_test, y_test, rocfile)

    confusionMat(model, x_test, y_test, cmfile, threshold)
    
    store(model, storepath)

    f1, acc = accuracy(model, x_test, y_test, threshold)

    varaibleImportances = importances(model, x_test, y_test).to_dict()

    storeIt('DLModel', description, {'f1score': f1, 'regularAccuracy': acc, 'AUC': area}, storepath, resultsFile, varaibleImportances)

if __name__ == '__main__':
    
    # create args object for command line operability
    args = parser.parse_args()
    
    # save folder name inputed for test data
    folder = args.input
    
    # open all data from folder and storing as a dataframe
    x_train = pd.read_csv(os.path.join(folder, 'x_train.csv'))
    x_test = pd.read_csv(os.path.join(folder, 'x_test.csv'))
    y_train = pd.read_csv(os.path.join(folder, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(folder, 'y_test.csv'))

    # number of inputs into the model stored for later use
    iSize = len(x_train.columns)

    # train and quickly evaluate model
    if args.tuninghyperparameters != None:
        with open(args.tuninghyperparameters) as file:
            d = json.load(file)
        model, params = hyperparameterTuning(createIt, x_train, y_train, d, iSize)
        evaluate(model, x_test, y_test, args.results, args.output, args.matrix, args.curve, description=params)

    else:
        # create a model shell
        model = createIt(iSize)
        train(model, x_train, y_train, x_test, y_test)
        evaluate(model, x_test, y_test, args.results, args.output, args.matrix, args.curve)

