# Import the necessary module
import csv

# Define a function called 'storeIt that takes in a name, description, metric, path, and results path as inputs
def storeIt(name, description, metric, path, resultsPath, importances=None):
    # Define a list of field names to use as a column headers for the csv file
    fieldnames = ['Name', 'Description', 'AUC', 'F1score', 'Accuracy', 'Path', 'Importances']
    # Define a dictionary containing the data to store in the csv file
    dictionary = {'Name': name, 'Description': description, 'AUC': metric['AUC'], 'F1score': metric['f1score'], 'Accuracy': metric["regularAccuracy"], 'Path': path, 'Importances': importances}

# Open the results path in append mode, and use the csv module to write the dictionary to the file    
    with open(resultsPath, 'a') as file:
        obj = csv.DictWriter(file, fieldnames=fieldnames)
        obj.writerow(dictionary)
# Print a message indicating that the entry was stored successfully
    print(f'Entry stored to {resultsPath}')
