import csv

def store(name, description, metric, path, resultsPath='./results.csv'):
    
    fieldnames = ['Name', 'Description', 'Metric', 'Path']
    dictionary = {'Name': name, 'Description': description, 'Metric': metric, 'Path': path}

    with open(resultsPath, 'a') as file:
        obj = csv.DictWriter(file, fieldnames=fieldnames)
        obj.writerow(dictionary)

    print(f'Entry stored to {resultsPath}')
