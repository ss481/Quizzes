import pandas as pd
import io
import requests
import numpy

from sklearn.preprocessing import LabelEncoder

#Read Train file from URL
url="http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
s=requests.get(url).content
data=pd.read_csv(io.StringIO(s.decode('utf-8')), header = None)

#Convert all strings
le = LabelEncoder()
for column in data.columns:
   	if data[column].dtype == type(object):
        	data[column] = le.fit_transform(data[column].astype(str))

features = []
target = []
i = 0
for c in data.columns:
	if i == len(data.columns)-1:
		target.append(c)
		#print(target)
	else:
		features.append(c)
		#print(features)
		i = i + 1

X_train = data[features]
y_train = data[target]


#Read Test file from URL
url="http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
s=requests.get(url).content
data=pd.read_csv(io.StringIO(s.decode('utf-8')),  skiprows=1, header = None)

#Convert all strings
for column in data.columns:
   	if data[column].dtype == type(object):
        	data[column] = le.fit_transform(data[column].astype(str))

features = []
target = []
i = 0
for c in data.columns:
	if i == len(data.columns)-1:
		target.append(c)
		#print(target)
	else:
		features.append(c)
		#print(features)
		i = i + 1


X_test = data[features]
y_test = data[target]

###################################################################################################
#Exploring data

num_training = len(y_train)
num_attributes = len(data.columns)-1
num_classes = len(pd.unique(y_train.values.ravel()))

ratio_sample_attributes = num_attributes / num_training
ratio_classes_attributes = num_classes / num_attributes

if num_classes > 2:
	type_classification = 1 #Multy class
else:
	type_classification = 0 #Binary


import collections
c = collections.Counter(y_train[target[0]])

ar = numpy.array([])
for x in c:
	key = x 
	value = c[key]	
	ar = numpy.append(ar, value/len(y_train[target[0]]))
	
import scipy as sc
entropy = sc.stats.entropy(ar, base = 2.0)

print('Number of training samples: ' + str(num_training))
print('Number of attributes: ' + str(num_attributes))
print('Number of classes: ' + str(num_classes))

print('Ratio between traing samples and attributes: ' + str(ratio_sample_attributes))
print('Ratio between classes and attributes: ' + str(ratio_classes_attributes))

print('Type classification: ' + str(type_classification))

print('Entropy: ', entropy)

print('MODELS: ')
####################################################################################################
#Linear models

#SVM Linear
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

lsvm = LinearSVC()
lsvm.fit(X_train, y_train.values.ravel())
pLsvm = lsvm.predict(X_test)
print ('Linear SVM: ' + str(accuracy_score(y_test, pLsvm)))

#Perceptron
from sklearn.linear_model import Perceptron

per = Perceptron()
per.fit(X_train, y_train.values.ravel())
pPer = per.predict(X_test)
print ('Perceptron: ' + str(accuracy_score(y_test, pPer)))


######################################################################################################
#Bonus models

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(DecisionTreeClassifier(),algorithm="SAMME", n_estimators=200)
ada.fit(X_train, y_train.values.ravel())
pAda = ada.predict(X_test)
print ('Ada Boost Decision Trees: ' + str(accuracy_score(y_test, pAda)))

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(weights = 'uniform')
knn.fit(X_train, y_train.values.ravel())
pKnn = knn.predict(X_test)
print ('KNN with uniform: ' + str(accuracy_score(y_test, pKnn)))

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train.values.ravel())
pNb = nb.predict(X_test)
print ('Naive bayes: ' + str(accuracy_score(y_test, pNb)))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators = 500, max_features = 10,criterion = 'gini')
rf.fit(X_train, y_train.values.ravel())
pRf = rf.predict(X_test)
print ('Random Forest: ' + str(accuracy_score(y_test, pRf)))

#########################################################################################################
