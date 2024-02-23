import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
print(dataset) 

#Dimensions of Data

print(dataset.shape)

#Peak the Data

print(dataset.head(20))

#Statistical Summary

print(dataset.describe())

#Class Distribution

print(dataset.groupby('CRIM').size()) #1
print(dataset.groupby('ZN').size()) #2
print(dataset.groupby('INDUS').size()) #3
print(dataset.groupby('CHAS').size()) #4
print(dataset.groupby('NOX').size()) #5
print(dataset.groupby('RM').size()) #6
print(dataset.groupby('AGE').size()) #7
print(dataset.groupby('DIS').size()) #8
print(dataset.groupby('RAD').size()) #9
print(dataset.groupby('TAX').size()) #10
print(dataset.groupby('PTRATIO').size()) #11
print(dataset.groupby('B').size()) #12
print(dataset.groupby('LSTAT').size()) #13
print(dataset.groupby('MEDV').size()) #14

#Univariate Plots

CRIM = dataset.iloc[:,0:1]
ZN = dataset.iloc[:,1:2]
INDUS = dataset.iloc[:,2:3]
CHAS = dataset.iloc[:,3:4]
NOX = dataset.iloc[:,4:5]
RM = dataset.iloc[:,5:6]
AGE = dataset.iloc[:,6:7]
DIS = dataset.iloc[:,7:8]
RAD = dataset.iloc[:,8:9]
TAX = dataset.iloc[:,9:10]
PTRATIO = dataset.iloc[:,10:11]
B = dataset.iloc[:,11:12]
LSTAT = dataset.iloc[:,12:13]
MEDV = dataset.iloc[:,13:14]

#1
CRIM.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#2
ZN.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#3
INDUS.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#4
CHAS.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#5
NOX.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#6
RM.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#7
AGE.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#8
DIS.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#9
RAD.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#10
TAX.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#11
PTRATIO.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#12
B.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#13
LSTAT.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#14
MEDV.plot(kind='box', subplots=True, layout=(1,1), sharex=False, sharey=False)
plt.show()

#Histograms

dataset.hist()
plt.show()

#Scatter Plot Matrix

scatter_matrix(dataset)
plt.show()

#Create a Validation Dataset

array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#Evaluate Each Model In Turn

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#Compare Algorithms

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()