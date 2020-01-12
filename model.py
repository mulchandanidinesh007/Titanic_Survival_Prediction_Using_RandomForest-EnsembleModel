#Tittanic Dataset: It is your job to predict if a passenger survived the sinking of the Titanic or not.
#For each in the test set, you must predict a 0 or 1 value for the variable.

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7,9]].values #Independent vector
Y = dataset.iloc[:, 1].values   #Dependent vector

#Handling Missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, [2]])
X[:, [2]] = imputer.transform(X[:, [2]])

#Encode Categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Random Forest Classification  to Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

#Predicting the results
Y_pred = classifier.predict(X_test)

#Making the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Calculating Precision and F1 score
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(Y_test, Y_pred)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))

from sklearn.metrics import recall_score
recall_score(Y_test, Y_pred, average='macro')

from sklearn.metrics import f1_score
f1_score(Y_test, Y_pred, average='macro')



