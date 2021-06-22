'''Author: Srinivasaraghavan Seshadhri, MSc Artificial Student, Cork Institute of Technology, R00195470
This is Practical Machine Learning Assignment 1 Part 1A
'''

"""
The objective is to build a KNN regression algorithm from scratch according to the given specifications
in the assignment.
NOTE: I am a fan of Scikit's I/O format and would like to mimic the format a little bit,
with own customization. The specifications are followed from assignment with a little bit extra added.

However, I have also attached the assignment without any changes in the bottom if this is not acceptable.

Please note that this code is not developed to production level yet. It would become into that level when some
more essential checks are done, warnings are given, error handling is done more effectively,
accepting prediction of 2D data and more adding features, and ofcourse lots of testing and debugging.

This program has been highly optimized to be as fast as possible, by implementing the fastest functions like
numpy are array based operations, use of 'map()' function has been implemented instead of loops for
faster repeated operations. There are NO for-loops in this entire program.
"""

#Importing required Libraries, as minimal as possible
import numpy as np
from matplotlib import pyplot as plt
from time import time


#All the specified functions has been put into the below class
#Class defined for K Nearest Neighbour Regression
class kNN_Regression:
    #The init function initializes reqd variables. It is done in such a way so that required data can be obtained later
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.k = 3
        self.n = 2
        self.z = None
        self.time_taken = None

    #Specified function calculates the Euclidean distance between a data and all given datas
    def calculate_distances(self,features,test_point):
        return np.sqrt(np.sum(np.square(features-test_point), axis=1))#Eucledian distance
    
    #Specified function returns the predicted target for a single data
    def predict(self,query,training_data = None):
        if len(query.shape)>1:
            raise ValueError(f"The query parameter must have a 1D data, but passed data has {query.shape} dimension(s)")
        if training_data is None:
            X_ = self.X_train
            y_ = self.y_train
        else:
            X_ = training_data[:,:training_data.shape[1]-1]
            y_ = training_data[:,training_data.shape[1]-1]
        #If an identical record exists in training data as query, it returns the corresponding target value as prediction.
        if query in X_:
            return y_[np.all(X_ == query, axis=1)]
        dist = self.calculate_distances(X_,query)
        #The following few lines of code can be reduced, without using induvidual variables, it has been left it as such
        #for the ease of reading
        critical_points = np.argsort(dist)[:self.k]#Closest k values
        critical_distances = dist[critical_points]#Respective distances from point to k closest values
        critical_values = y_[critical_points]#Respective target values
        weights = critical_distances**(-self.n)#Weight is the inverse of distance powered n times
        return np.sum(np.multiply(critical_values,weights))/ np.sum(weights)#Predicted target value
    
    #Specified function calculates and returns the R^2 score for given actual and predicted target values
    def calculate_r2(self,y, y_hat):
        if y.shape != y_hat.shape:#Checks and raises error if the input dimensions aren't correct/matching
            raise ValueError(f"Both the input parameters must match, but yours is {y.shape} for y and {y_hat.shape} for y_hat")
        return 1-(np.sum(np.square(y_hat-y))/np.sum(np.square(np.mean(y_hat)-y)))#Returns R^2 score
    
    #Own function mimicing sklearn's standards. Note, sk learn's fit doesn't accept the hyperparameters.
    #We are accepting hyperparameters here for convinience purpose and to show a difference
    def fit(self,X_train,y_train,k=3,n=2):
        self.k = k
        self.n = n
        self.train_data = np.append(X_train,np.array(y_train.reshape(-1,1)),axis=1)
        self.X_train = X_train
        self.y_train = y_train
        return f'Fit Successful with X_train,y_train,k={self.k},n={self.n}'
        
    #Own function to easily calculate the R^2 score for any test data 
    def score(self,X_test,y_test):
        self.test_data = np.append(X_test,np.array(y_test.reshape(-1,1)),axis=1)
        self.X_test = X_test
        self.y_test = y_test
        start = time()
        z = np.array(list(map(self.predict, self.X_test, [self.train_data]*self.test_data.shape[0]))).flatten()
        r2 = self.calculate_r2(self.y_test, z)
        self.time_taken = time()-start
        self.z = z
        return r2
    
    #Own function to plot graph between actual and predicted target values
    #Dots forming straighter and thinner line shows high accuracy
    #Dots formed, the wider area they cover, the more inaccurate they are
    def visualize(self, marker_ = '.', s_ = 1):
        plt.scatter(self.y_test, self.z, marker = marker_, s=s_)
        plt.ylabel('Original Values')
        plt.xlabel("Predicted Values")
        plt.show()
        
    #Own function Display the time taken by the 'score' function to predict the targets and calculate r^2 score.
    def duration(self):
        print("Time taken:",self.time_taken,"secs")


#The following reads data from the given CSV files and assigns them in appropriate variables
train_data = np.genfromtxt('trainingData_reg.csv', dtype = 'float', delimiter = ',')
test_data = np.genfromtxt('testData_reg.csv', dtype = 'float', delimiter = ',')
X_train = train_data[:,:12]
y_train = train_data[:,12]
X_test = test_data[:,:12]
y_test = test_data[:,12]

#Instanciating the class
knn = kNN_Regression()

#Passing in the training data
print("Fit:",knn.fit(X_train,y_train))

#Passing in the test data to predict the targets and check its accuracy, displaying R^2 score
print("R^2 Score:",knn.score(X_test,y_test))


#Passing in a single data feature to predict and comparing with the actual value
print(f"""Prediction for X_test[0]:{knn.predict(X_test[0])}
whereas Original value: {y_test[0]}""")

#Displaying the time taken by the 'score' function to predict the targets and calculate r^2 score.
knn.duration()

#Plotting graph between actual and predicted target values
knn.visualize()




'''
Here is the code in the exact specifications according to assignment, without using classes, extra functions etc.

If you wish to run the following, please comment the above code, uncomment the below code and run the same.
'''
'''
import numpy as np
from matplotlib import pyplot as plt
from time import time

#Specified function calculates the Euclidean distance between a data and all given datas
def calculate_distances(features,test_point):
        return np.sqrt(np.sum(np.square(features-test_point), axis=1))

#Specified function returns the predicted target for a single data
def predict(training_data,query,k=3,n=2):
    X_ = training_data[:,:training_data.shape[1]-1]
    y_ = training_data[:,training_data.shape[1]-1]
    if query in X_:
        return y_[np.all(X_ == query, axis=1)]
    dist = calculate_distances(X_,query)
    critical_points = np.argsort(dist)[:k]
    critical_distances = dist[critical_points]
    critical_values = y_[critical_points]
    weights = critical_distances**(-n)
    return np.sum(np.multiply(critical_values,weights))/ np.sum(weights)

#Specified function calculates and returns the R^2 score for given actual and predicted target values
def calculate_r2(y, y_hat):
    if y.shape != y_hat.shape:
        raise ValueError(f"Both the input parameters must match, but yours is {y.shape} for y and {y_hat.shape} for y_hat")
    ssr = np.sum(np.square(y_hat-y))
    tss = np.sum(np.square(np.mean(y_hat)-y))
    return 1-ssr/tss


#The following reads data from the given CSV files and assigns them in appropriate variables
train_data = np.genfromtxt('trainingData_reg.csv', dtype = 'float', delimiter = ',')
test_data = np.genfromtxt('testData_reg.csv', dtype = 'float', delimiter = ',')
X_train = train_data[:,:12]
y_train = train_data[:,12]
X_test = test_data[:,:12]
y_test = test_data[:,12]


start = time()

#Collecting all predicted test targets
z = np.array(list(map(predict, [train_data]*test_data.shape[0], X_test))).flatten()

#Calculate the R^2 score
r2 = calculate_r2(y_test, z)

#Display the requied details 
print(f'Time taken: {time()-start} secs')
print(f'R^2 score: {r2}')

#Plotting graph between actual and predicted target values
plt.ylabel('Original Values')
plt.xlabel("Predicted Values")
plt.scatter(y_test, z, marker = '.', s=1)
plt.show()
'''