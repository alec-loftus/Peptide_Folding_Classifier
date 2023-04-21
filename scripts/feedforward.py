# import necessary libraries
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import pandas as pd
import numpy as np
import argparse
from accuracy import confusionMat
import os

# creating command line parser for flags to run script from the command line in a customized manner
parser = argparse.ArgumentParser(description="train and evaluate a feed forward fully connected neural network")

# input folder path flag created
parser.add_argument('-i', '--input', help='input folder with data (should include scaled/normalized x_train, x_test, y_test, y_train CSVs', required=True)

# output file name flag created
parser.add_argument('-o', '--output', help='output file for model object; must be pkl', required=False, default='./output.pkl')

# results.csv path flag created
parser.add_argument('-r', '--results', help='path to results.csv (must have Name, Description, Metric, Path)', required=True)

# matrix path flag created
parser.add_argument('-m', '--matrix', help='path to matrix file; must be png', required=False, default='./output.png')

# customized 1 set of hyperparameters
parser.add_argument('-s', '--startinghyperparameters', help='json file with hyperparameters for customized model training (update with types that can be played around with)', required=False, default=None)

# list of options of hyperparameters for tuning
parser.add_argument('-t', '--tuninghyperparameters', help='json file with lists of hyperparameters for each option (update with types that can be played around with)', required=False, default=None)


def create(inputSize, numHiddenLayers=1, numHiddenNodes=None, activationHidden='relu'):
    '''
    Creates model layers and assorts them together
    '''
    
    if numHiddenNodes == None:
        numHiddenNodes = inputSize+4

    i = Input(shape=(inputSize,)) 
    l1 = Dense(numHiddenNodes, activation=activationHidden)(i)
    listLayers = [l1]
    for i in range(numHiddenLayers-1):
        l = Dense(numHiddenNodes, activation=activationHidden)(listLayers[-1])
        listLayers.append(l)
    o = Dense(1, activation='softmax')(listLayers[-1])
    
    model = Model(inputs=i, outputs=o)

    # returns overview of model shell
    model.summary()

    return model

def train(model, x_train, y_train, x_test, y_test, optimizer='adam', loss='mse', epochs=1000, batch_size=40):
    '''
    Trains model on training data that is stored in the generator
    '''
    
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    predictions, y_test = evaluate(model, 0.5, x_test, y_test)
    print(predictions)
    print(y_test)

def hyperparameterTuning():
    '''
    Selects best model from a park of models with a varied combindation of hyperparameters as specified by the user
    '''
    return

def evaluate(model, threshold, x_test, y_test, saveFile='output.json'):
    '''
    Evaluates trained model on test data set and stores results
    '''

    predictions = model.predict(x_test)
    # predictions = np.where(predictions<threshold,1,0)

    return predictions, y_test


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

    # create a model shell
    model = create(iSize)
    
    # train and quickly evaluate model
    train(model, x_train, y_train, x_test, y_test)

    # evaluate model performance 
