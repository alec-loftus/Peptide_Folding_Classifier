#import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
from glob import glob
import os
import pdb

#set up command line argument parser
parser = argparse.ArgumentParser(description='Split data into test/train')
parser.add_argument('-i', '--input', help='input path to data folder', required=True)
parser.add_argument('-o', '--output', help='output folder path for test/train csvs', required=False, default='./output')
parser.add_argument('-s', '--splitpercent', help='percentage of data to store as test', required=False, default='30')
parser.add_argument('-t', '--threshold', help='scoring threshold', required=False, default=None)
parser.add_argument('-d', '--data', help='list of column names of data to input', nargs='*', required=False, default=None)
parser.add_argument('-p', '--predictingColumn', help='name of column to predict off of', required=True)
parser.add_argument('-n', '--newLabel', help='new label for scoring column', required=False, default='label')

#combine all input csv files into one dataframe
def combine(files):
    dataframes = []
    finalData = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    finalData = pd.concat(dataframes)

    return finalData

#add a new column to the data frame with values based on a threshold
def score(treshold, column, data, name):
    data[name]=np.where(data[column]<treshold,True,False)

#split the data frame into training and test sets and save them as csv files
def split(data, labelName, percent, output, inputLabels=None):
    y = data[labelName]
    x = data.drop(labelName, axis=1)
    if inputLabels != None:
        x = data[inputLabels]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent/float(100), random_state=42, shuffle=True)
    
    os.mkdir(output)
    
    scaler = MinMaxScaler()

    x_train.to_csv(os.path.join(output, 'x_train.csv'), index=False)
    print(f'x_train.csv saved to {output}!')
    y_train.to_csv(os.path.join(output, 'y_train.csv'), index=False)
    print(f'y_train.csv saved to {output}!')
    x_test.to_csv(os.path.join(output, 'x_test.csv'), index=False)
    print(f'x_test.csv saved to {output}!')
    y_test.to_csv(os.path.join(output, 'y_test.csv'), index=False)
    print(f'y_test.csv saved to {output}!')

if __name__ == '__main__':
    args = parser.parse_args()
   
#find all csv files in the input folder and combine them into one dataframe
    path = os.path.join(args.input, '*.csv')
    listOFiles = glob(path)
    
    df1 = combine(listOFiles)
    
#print the column names of the combined dataframe    
    print(df1.columns)

#apply a threshold to the label column if a threshold value was provided and save the new column as 'isFolded'    
    labelColumn = args.predictingColumn
    if args.threshold != None:
        score(float(args.threshold), labelColumn, df1, args.newLabel)
        print(df1.columns)
        df1 = df1.drop(labelColumn, axis=1)
        labelColumn = args.newLabel
#split the dataframe into training and test sets and save them as csv files
    if args.data == None:
        split(df1, labelColumn, float(args.splitpercent), args.output)
    else:
        split(df1, labelColumn, float(args.splitpercent), args.output, inputLabels=list(args.data))
