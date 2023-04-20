# Import necessary module
import pickle

# Define a function called store that takes in a trained model and a file path as inputs
def store(model, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f'Model stored to the following path: {filepath}')

# Define a function called 'openIt' that takes a file path as input    
def openIt(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

