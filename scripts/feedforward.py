from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description="train and evaluate a feed forward fully connected neural network")
parser.add_argument('-i', '--input', help='input folder with data (should include scaled/normalized x_train, x_test, y_test, y_train CSVs', required=True)
parser.add_argument('-o', '--output', help='output file for model object; must be pkl', required=False, default=./output.pkl)
parser.add_argument('-r', '--results', help='path to results.csv (must have Name, Description, Metric, Path)', required=True)
parser.add_argument('-m', '--matrix', help='path to matrix file; must be png', required=False, default='./output.png')



def create():
    return

def train()
    return

def evaluate():
    return


if __name__ == '__main__':
    
    folder = args.input
    
    x_train = pd.read_csv(os.path.join(folder, 'x_train.csv'))
    x_test = pd.read_csv(os.path.join(folder, 'x_test.csv'))
    y_train = pd.read_csv(os.path.join(folder, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(folder, 'y_test.csv'))

    
