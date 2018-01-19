# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('criminal_train.csv')
X = dataset.iloc[:, 1:71].values
y = dataset.iloc[:, 71].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, y)

# Importing the dataset
dataset1 = pd.read_csv('criminal_test.csv')
X_test = dataset1.iloc[:, 1:].values
sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


