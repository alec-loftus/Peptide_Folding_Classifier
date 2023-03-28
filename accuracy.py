import sklearn
from sklearn.metrics import accuracy_score
#will be added in with data previously
def accuracy(y_pred, y_true):
  return accuracy_score(y_true, y_pred)
