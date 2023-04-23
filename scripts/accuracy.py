# Import necessary libraries
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np

# Define a function to calculate accuracy of a model
def accuracy(model, x_test,  y_test, threshold=0.5):
    # Make predictions on the test features using the model
    try:
        y_pred = list(model.predict_proba(x_test))
        y_pred = np.array([y[1] for y in y_pred])
    except:
        y_pred = list(model.predict(x_test).ravel())
    # convert prediction probabilities to integers
    y_pred = [1 if float(y)>=threshold else 0 for y in y_pred]
    # Calculate the f1 score of the predictions with respect to the true test labels
    return f1_score(y_test, y_pred), accuracy_score(y_test, y_pred)

# Define a function to create and store a confusion matrix of a model's predictions
def confusionMat(model, x_test, y_test, storeFile, threshold=0.5):
    # Make predictions on the test features using the model
    try:
        y_pred = list(model.predict_proba(x_test))
        y_pred = np.array([y[1] for y in y_pred])
    except:
        y_pred = list(model.predict(x_test).ravel())
    # convert prediction probabilities to integers
    y_pred = [1 if float(y)>=threshold else 0 for y in y_pred]
    # Calculate the confusion matrix of the predictions with respect to the true test labels
    arr = confusion_matrix(y_test, y_pred)
    # Create a figure and axis object
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Create a heatmap of the confusion matrix
    cax = ax.matshow(arr)
     # Set the title and colorbar of the figure
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    # Set the x and y labels of the axis
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add the values of the confusion matrix to the heatmap
    for (i, j), z in np.ndenumerate(arr):
        ax.text(j, i, z, ha='center', va='center')
    
     # Save the figure to a file specified by the storeFile argument
    with open(storeFile, 'w') as file:
        fig.savefig(storeFile)
        # Print a message indicating the location of the saved file
        print(f'Confusion Matrix Stored at {storeFile}!')

# plot roc_curve
def roc(model, x_test, y_test, storeFile):
    try:
        y_pred = list(model.predict_proba(x_test))
        y_pred = np.array([y[1] for y in y_pred])
    except:
        y_pred = list(model.predict(x_test).ravel())
        print('non-proba')
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, drop_intermediate=False) # fpr, tpr and thresholds for roc curve
    area = auc(fpr, tpr)
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    threshold = round(thresholds[index], ndigits = 4)

    # plot and save ROC curve
    fig = plt.figure
    plt.plot([0, 1], [0, 1], 'k--') # plots random guess line
    plt.plot(fpr, tpr, label=f'AUC = {area}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(storeFile)
    print(f'ROC curve saved to {storeFile}')
    return area, threshold

