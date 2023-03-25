import pandas as pd
import sklearn
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser(description='Split data into test/train')
parser.add_argument('-i', '--input', help='input path to data folder', required=True)
parser.add_argument('-o', '--output', help='output folder path for test/train csvs', required=False, default='./output')
parser.add_argument('-s', '--splitpercent', help='percentage of data to store as test', required=False, default='30')
parser.add_argument('-l', '--label', help='name of data label column', required=False, default='isFolded')
parser.add_argument('-t', '--treshold', help='scoring threshold', required=False, default=None)
parser.add_argument('-j', '--judgement', help='column to judge off of', required=False, default='RgEnd')

def combine(files):
    dataframes = []
    finalData = []
    for file in files:
        df = pd.read_table(file)
        dataframes.append(df)

    finalData = dataframes[0]

    for df in (1, dataframes):
        finalData = finalData.append(df)

    return finalData

def score(treshold, column, data, labelName):
    data[labelName]=np.where(df[column]<treshold,True,False)

def split(data, labelName, percent, output):
    y = data[labelName]
    x = data.drop(labelName)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=percent/float(100), random_state=42, shuffle=True)
    x_train.to_csv(os.path.join(output, 'x_train.csv'))
    print(f'x_train.csv saved to {output}!')
    y_train.to_csv(os.path.join(output, 'y_train.csv'))
    print(f'y_train.csv saved to {output}!')
    x_test.to_csv(os.path.join(output, 'x_test.csv'))
    print(f'x_test.csv saved to {output}!')
    y_test.to_csv(os.path.join(output, 'y_test.csv'))
    print(f'y_test.csv saved to {output}!')

if __name__ == '__main__':
    args = parser.parse_args()
    
    path = os.path.join(args.input, '*.csv')
    listOfiles = glob(path)
    
    df = combine(listOfFiles)

    if args.threshold != None:
        df = score(args.threshold, args.judgement, df, args.label)
        df = df.drop(args.judgement)

    split(df, args.label, args.splitpercent, args.output)
