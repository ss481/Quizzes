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


##########################################################################################
#Visualize data

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


#Male vs Female

n_groups = 2
maleH = 0
maleL = 0
femaleH = 0
femaleL = 0

for i in range(0,len(X_train)):
	if y_train[14][i] == 1 and X_train[9][i] == 1:
		maleH = maleH + 1 
	elif y_train[14][i] == 0 and X_train[9][i] == 1:
		maleL = maleL + 1
	elif y_train[14][i] == 1 and X_train[9][i] == 0:
		femaleH = femaleH + 1
	elif y_train[14][i] == 0 and X_train[9][i] == 0:
		femaleL = femaleL + 1

high = (maleH, femaleH)
low = (maleL, femaleL)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, high, bar_width,
                 alpha=opacity,
                 color='b',
                 label='>50K')
 
rects2 = plt.bar(index + bar_width, low, bar_width,
                 alpha=opacity,
                 color='g',
                 label='<=50K')


plt.xlabel('Number of Instances')
plt.ylabel('Sex')
plt.title('Salary by Sex')
plt.xticks(index + bar_width, ('Male',  'Female'))
plt.legend()
 
plt.tight_layout()
plt.show()



#Maritial status
n_groups = 7

DivH = 0
DivL = 0

mafH = 0
mafL = 0

mcivH = 0
mcivL = 0

msaH = 0
msaL = 0

nmH = 0
nmL = 0

sH = 0
sL = 0

wH = 0
wL = 0

for i in range(0,len(X_train)):
	if y_train[14][i] == 1 and X_train[5][i] == 0:
		DivH = DivH + 1 
	elif y_train[14][i] == 0 and X_train[5][i] == 0:
		DivL = DivL + 1
	elif y_train[14][i] == 1 and X_train[5][i] == 1:
		mafH = mafH + 1
	elif y_train[14][i] == 0 and X_train[5][i] == 1:
		mafL = mafL + 1	
	elif y_train[14][i] == 1 and X_train[5][i] == 2:
		mcivH = mcivH + 1
	elif y_train[14][i] == 0 and X_train[5][i] == 2:
		mcivL = mcivL + 1
	elif y_train[14][i] == 1 and X_train[5][i] == 3:
		msaH = msaH + 1
	elif y_train[14][i] == 0 and X_train[5][i] == 3:
		msaL = msaL + 1
	elif y_train[14][i] == 1 and X_train[5][i] == 4:
		nmH = nmH + 1
	elif y_train[14][i] == 0 and X_train[5][i] == 4:
		nmL = nmL + 1
	elif y_train[14][i] == 1 and X_train[5][i] == 5:
		sH = sH + 1
	elif y_train[14][i] == 0 and X_train[5][i] == 5:
		sL = sL + 1
	elif y_train[14][i] == 1 and X_train[5][i] == 6:
		wH = wH + 1
	elif y_train[14][i] == 0 and X_train[5][i] == 6:
		wL = wL + 1
high = (DivH, mafH, mcivH, msaH, nmH, sH, wH)
low = (DivL, mafL, mcivL, msaL, nmL, sL, wL)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.barh(index, high, bar_width,
                 alpha=opacity,
                 color='b',
                 label='>50K')
 
rects2 = plt.barh(index + bar_width, low, bar_width,
                 alpha=opacity,
                 color='g',
                 label='<=50K')


	
plt.xlabel('Number of Instances')
plt.ylabel('Maritial Status')
plt.title('Salary by Maritial Status')
plt.yticks(index + bar_width, ('Divorced',  'Married-AF-spouse', 'Married-civ-spouse',  'Married-spouse-absent', 'Never-married', 'Separated','Widowed'))
plt.legend()
 
plt.tight_layout()
plt.show()

#Education

n_groups = 16

tenthH = 0
tenthL = 0
eleventhH = 0
eleventhL = 0

twelvethH = 0
twelvethL = 0

fifthH = 0
fifthL = 0

onethH = 0
onethL = 0

seventhH = 0
seventhL = 0

ninethH = 0
ninethL = 0


assacH = 0
assacL = 0

assvoH = 0
assvoL = 0

bscH = 0
bscL = 0

docH = 0
docL = 0

hsH = 0
hsL = 0

mscH = 0
mscL = 0

preH = 0
preL = 0

profH = 0
profL = 0

sH = 0
sL = 0
for i in range(0,len(X_train)):
	if y_train[14][i] == 1 and X_train[3][i] == 0:
		tenthH = tenthH + 1 
	elif y_train[14][i] == 0 and X_train[3][i] == 0:
		tenthL = tenthL + 1
	elif y_train[14][i] == 1 and X_train[3][i] == 1:
		eleventhH = eleventhH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 1:
		eleventhL = eleventhL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 2:
		twelvethH = twelvethH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 2:
		twelvethL = twelvethL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 3:
		fifthH = fifthH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 3:
		fifthL = fifthL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 4:
		onethH = onethH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 4:
		onethL = onethL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 5:
		seventhH = seventhH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 5:
		seventhL = seventhL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 6:
		ninethH = ninethH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 6:
		ninethL = ninethL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 7:
		assacH = assacH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 7:
		assacL = assacL + 1
	elif y_train[14][i] == 1 and X_train[3][i] == 8:
		assvoH = assvoH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 8:
		assvoL = assvoL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 9:
		bscH = bscH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 9:
		bscL = bscL + 1
	elif y_train[14][i] == 1 and X_train[3][i] == 10:
		docH = docH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 10:
		docL = docL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 11:
		hsH = hsH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 11:
		hsL = hsL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 12:
		mscH = mscH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 12:
		mscL = mscL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 13:
		preH = preH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 13:
		preL = preL + 1
	elif y_train[14][i] == 1 and X_train[3][i] == 14:
		profH = profH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 14:
		profL = profL + 1	
	elif y_train[14][i] == 1 and X_train[3][i] == 15:
		sH = sH + 1
	elif y_train[14][i] == 0 and X_train[3][i] == 15:
		sL = sL + 1			

	

	
high = (tenthH, eleventhH, twelvethH, fifthH, onethH, seventhH, ninethH, assacH, assvoH, bscH, docH, hsH, mscH, preH, profH, sH)
low = (tenthL, eleventhL, twelvethL, fifthL, onethL, seventhL, ninethL, assacL, assvoL, bscL, docL, hsL, mscL, preL, profL, sL)
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.barh(index, high, bar_width,
                 alpha=opacity,
                 color='b',
                 label='>50K')
 
rects2 = plt.barh(index + bar_width, low, bar_width,
                 alpha=opacity,
                 color='g',
                 label='<=50K')


	
plt.xlabel('Number of Instances')
plt.ylabel('Education')
plt.title('Salary by Education')
plt.yticks(index + bar_width, ('10th Grade',  '11th Grade', '12th Grade',  '5-6th Grade', '1-4th Grade', '7-8th Grade','9th Grade', 'Assoc-acdm','Assoc-voc', 'Bachelor', 'Doctorate', 'High school grad', 'Master','Preschool','Prof - school', 'Some college'))
plt.legend()
 
plt.tight_layout()
plt.show()