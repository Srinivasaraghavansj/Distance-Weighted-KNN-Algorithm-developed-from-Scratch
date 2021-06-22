 """
Author: Srinivasaraghavan Seshadhri, MSc Artificial Intelligence student, Cork Institute of Technology, R00195470
This is Practical Machine Learning Assignment 1 Part 2A
"""

"""
The objective is to compare the effect of hyperparameters in KNN Regression algorithm on the given dataset
and SK-learn's Boston house pricing dataset and save in a excel file. This is a one time run code only and hence the coding
efficiency doesn't matter. The code was run twice with different data, which is now commented out.
If you wish to try out the other dataset, please uncomment the commented code and comment the
other equivalent code.
"""
from sklearn.datasets import load_boston
import pandas as pd
from time import time
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# datas = load_boston()
# X = datas.data
# y = datas.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = np.genfromtxt('trainingData_reg.csv', dtype = 'float', delimiter = ',')
test_data = np.genfromtxt('testData_reg.csv', dtype = 'float', delimiter = ',')
X_train = train_data[:,:12]
y_train = train_data[:,12]
X_test = test_data[:,:12]
y_test = test_data[:,12]

diction = {
    'n_neighbours':[],
    'weights':[],
    'algorithm':[],
    'leaf_size':[],
    'score':[],
    'time':[]
}

n_neighbors=[1,3,5,7,10]
weights=['uniform','distance']
algorithms=['ball_tree','kd_tree','brute']
leaf_sizes=[15,30,60]

def runner(n_neighbor_, weight_, algorithm_, leaf_size_):
    neigh = KNeighborsRegressor(n_neighbors=n_neighbor_, weights=weight_, algorithm=algorithm_ , leaf_size = leaf_size_)
    start = time()
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test,y_test)
    time_ = time()-start
    neig = diction['n_neighbours']
    neig.append(n_neighbor_)
    diction['n_neighbours']=neig
    weig = diction['weights']
    weig.append(weight_)
    diction['weights']=weig
    algo=diction['algorithm']
    algo.append(algorithm_)
    diction['algorithm']=algo
    leaf = diction['leaf_size']
    leaf.append(leaf_size_)
    diction['leaf_size']=leaf
    scor = diction['score']
    scor.append(score)
    diction['score']=scor
    tim = diction['time']
    tim.append(time_)
    diction['time']=tim


for neighbor in n_neighbors:
        for weight in weights:
            for algorithm in algorithms:
                for leaf_size in leaf_sizes:
                    runner(neighbor,weight,algorithm,leaf_size)


df = pd.DataFrame(diction)

# df.to_excel('KNN_test_results.xlsx')
df.to_excel('KNN_test_results_Given.xlsx')