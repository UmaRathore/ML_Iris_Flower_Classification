
# Iris flower classification model - Machine Learning Project
# Skipping Step 1 : data gathering and Step 2: data cleaning, of Machine Learning,
# as this data set is downloaded from sklearn library and step 1 and step 2 are already executed

# Import Dataset

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print(target_names)
print(feature_names)

print("Class type of data X: ", type(X))

# Step 3: Splitting Data

# Split data available in scikit-learn library so that cleaned data can be used directly.
# There are 150 data points and after splitting them into train:test data sets, count reduces to 120:30

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)

import math
math.sqrt(len(y_test))

# Step 4: Model Selection

# Data model - supervised classification data model - K Nearest neighbors algorithm is used.
# There are three categories of IRIS flower in which data points can be classified, assuming k-neighbors = 3

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# Accuracy score is ~96
# Identify the best accuracy score for the different values of k
# Selection of value of k, as it should be odd and square root of data points.
# Train data sets are 120 in count, its sqrt is approx 11, so next set of values for k can be between 3-12

# Step 5: Analysis

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Calculating error and plotting graph for error against different values of k
# Values of k which has least error is probably the best solution

error_list = []
k_list = []
for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_list.append(np.mean(y_pred != y_test))
    k_list.append(k)

# graph to plot k-values vs error

import matplotlib.pyplot as plt
x = k_list
y = error_list
plt.plot(x, y)

plt.xlabel('k - values')
plt.ylabel('error')
plt.title('best value of k')
plt.show()

# Calculating accuracy and plotting graph for accuracy against different values of k
# Values of k which has maximum accuracy is probably the best solution

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
acc_list = []
k_list = []
for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    k_list.append(k)
    acc_list.append(accuracy)

# graph to plot k-values vs accuracy

x = k_list
y = acc_list
plt.plot(x, y)

plt.xlabel('k - values')
plt.ylabel('accuracy')
plt.title('best value of k')
plt.show()

# Step 6 : Improvement

# find the best parameters for a particular model, use GridSearchCV to find best value of k

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

knn_pipe = Pipeline([('mms', MinMaxScaler()),
                     ('knn', KNeighborsClassifier())])
params = [{'knn__n_neighbors': [1, 2, 3, 4, 5],
         'knn__weights': ['uniform', 'distance'],
         'knn__leaf_size': [3, 8]}]
gs_knn = GridSearchCV(knn_pipe,
                      param_grid=params,
                      scoring='accuracy',
                      cv=3)

gs_results = gs_knn.fit(X_train, y_train)

# for 5 values of k neighbors the best parameter value is 4 with a best model score of 0.95
# metrics accuracy is 1

print(gs_knn.best_params_)
print(gs_results.best_score_)

# calculating the metrics score with n_neighbors = 4

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))

# As value of k increases the accuracy score is improving thus considering, Next 5 set of values

params = [{'knn__n_neighbors': [5, 6, 7, 8, 9],
         'knn__weights': ['uniform', 'distance'],
         'knn__leaf_size': [3, 8]}]

gs_knn = GridSearchCV(knn_pipe,
                      param_grid=params,
                      scoring='accuracy',
                      cv=3)

gs_results = gs_knn.fit(X_train, y_train)
print(gs_knn.best_params_)
print(gs_results.best_score_)

# calculating the metrics score with n_neighbors = 5

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# As value of k increases the accuracy score is improving thus considering, Next 5 set of values
# Referring the results now almost score is getting constant thus,
# best performance is given by parameters between 1-12

print(metrics.accuracy_score(y_test, y_pred))

params = [{'knn__n_neighbors': [8, 9, 10, 11, 12],
         'knn__weights': ['uniform', 'distance'],
         'knn__leaf_size': [3, 8]}]
gs_knn = GridSearchCV(knn_pipe,
                      param_grid=params,
                      scoring='accuracy',
                      cv=3)

gs_results = gs_knn.fit(X_train, y_train)
print(gs_knn.best_params_)
print(gs_results.best_score_)

# calculating the metrics score with n_neighbors = 9

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

# heap map

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Creates a confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'],
                     columns = ['setosa','versicolor','virginica'])

sns.heatmap(cm_df, annot=True)
plt.title('Accuracy using brute:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()




