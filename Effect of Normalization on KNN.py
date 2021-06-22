 """
Author: Srinivasaraghavan Seshadhri, MSc Artificial Intelligence student, Cork Institute of Technology, R00195470
This is Practical Machine Learning Assignment 1 Part 2A
"""

"""
The objective is to compare the effect of Normalization and Scaling on the given dataset and SK-learn's
Boston house pricing dataset. This is a one time run code only and hence the coding efficiency doesn't
matter. It does contain repeatable tasks, but since it is only being repeated twice I decided to leave
it as such without putting them in a reusable function.
"""
from sklearn.datasets import load_boston
from time import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
def select_data(choice):
    if choice == 1:
        datas = load_boston()
        X = datas.data
        y = datas.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        train_data = np.genfromtxt('trainingData_reg.csv', dtype = 'float', delimiter = ',')
        test_data = np.genfromtxt('testData_reg.csv', dtype = 'float', delimiter = ',')
        X_train = train_data[:,:12]
        y_train = train_data[:,12]
        X_test = test_data[:,:12]
        y_test = test_data[:,12]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = select_data(0)
print("Given Data Selected")
neigh = KNeighborsRegressor(n_neighbors=5)
s = time()
neigh.fit(X_train, y_train)
print("Default score:",neigh.score(X_test,y_test))
print("Time Taken:",time()-s,'\n')
scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
neigh = KNeighborsRegressor(n_neighbors=5)
s = time()
neigh.fit(X_train, y_train)
print("Scaled Data's score:",neigh.score(X_test,y_test))
print("Time Taken:",time()-s,'\n')
X_train, X_test, y_train, y_test = select_data(0)
transformer = Normalizer().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
neigh = KNeighborsRegressor(n_neighbors=5)
s = time()
neigh.fit(X_train, y_train)
print("Normalized data's score:",neigh.score(X_test,y_test))
print("Time Taken:",time()-s,'\n')

X_train, X_test, y_train, y_test = select_data(1)
print("Boston Housing price Data Selected")
neigh = KNeighborsRegressor(n_neighbors=5)
s = time()
neigh.fit(X_train, y_train)
print("Default score:",neigh.score(X_test,y_test))
print("Time Taken:",time()-s,'\n')
scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
neigh = KNeighborsRegressor(n_neighbors=5)
s = time()
neigh.fit(X_train, y_train)
print("Scaled Data's score:",neigh.score(X_test,y_test))
print("Time Taken:",time()-s,'\n')
X_train, X_test, y_train, y_test = select_data(1)
transformer = Normalizer().fit(X_train)
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
neigh = KNeighborsRegressor(n_neighbors=5)
s = time()
neigh.fit(X_train, y_train)
print("Normalized data's score:",neigh.score(X_test,y_test))
print("Time Taken:",time()-s,'\n')