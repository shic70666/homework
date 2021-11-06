#!/usr/bin/env python
# coding: utf-8

''' Logistic Regression 逻辑回归
    Author: Hao Bai (hao.bai@insa-rouen.fr)
'''

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt



#!------------------------------------------------------------------------------
#!                                     CLASSES
#!------------------------------------------------------------------------------




#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------
def main(num_of_classess):
    ## Step 1: just import some data to play
    iris = datasets.load_iris()
    if num_of_classess == 2:
        X = iris.data[:100, :2] # only take the first 2 features
        Y = iris.target[:100]
    else:
        X = iris.data[:, :2] # only take the first 2 features
        Y = iris.target
    
    #* split the dataset into training dataset and testing dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    ## Step 2: train the model
    #* training the model
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    
    #* post-processing
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, label="Decision boundary")

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors="k", label="Training points")
    plt.xlabel("Feature 1: Sepal length")
    plt.ylabel("Feature 2: Sepal width")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()
    

    ## Step 3: use the model
    Y_predict = logreg.predict(X_test)

    #* post-processing
    plt.figure(1, figsize=(4, 3))
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="+", c=Y_test, s=50, cmap=plt.cm.Paired, label="True label")
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", c=Y_predict, s=50, cmap=plt.cm.Paired, label="Predicted label")
    plt.legend()
    plt.show()


#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
if __name__ == "__main__":
    main(num_of_classess=3)