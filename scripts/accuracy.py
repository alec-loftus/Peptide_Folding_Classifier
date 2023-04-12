from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#will be added in with data previously
def accuracy(model, x_test,  y_test):
    
    y_pred = model.predict(x_test)
    return f1_score(y_test, y_pred)

def confusionMat(model, x_test, y_test, storeFile):
    
    y_pred = model.predict(x_test)
    arr = confusion_matrix(y_test, y_pred)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(arr)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    with open(storeFile, 'w') as file:
        fig.savefig(storeFile)
        print(f'Confusion Matrix Stored at {storeFile}!')
