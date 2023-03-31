from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

#will be added in with data previously
def accuracy(y_pred, y_true):
  return accuracy_score(y_true, y_pred)

def confusionMat(model, x_test, y_test, storeFile):
  mat = plot_confusion_matrix(model, x_test, y_test)
  mat.xlabel('Prediction')
  mat.ylabel('True Label')
  with open(storeFile, 'w') as file:
    mat.savefig(storeFile)
    print(f'Confusion Matrix Stored at {storeFile}!')
