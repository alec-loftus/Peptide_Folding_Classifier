# Import necessary libraries
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Define a function to calculate accuracy of a model
def accuracy(model, x_test,  y_test):
    # Make predictions on the test features using the model 
    y_pred = model.predict(x_test)
    # Calculate the f1 score of the predictions with respect to the true test labels
    return f1_score(y_test, y_pred)
# Define a function to create and store a confusion matrix of a model's predictions
def confusionMat(model, x_test, y_test, storeFile):
    # Make predictions on the test features using the model
    y_pred = model.predict(x_test)
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
