import pickle

def store(model, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f'Model stored to the following path: {filepath}')

def open(filepath):
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model

