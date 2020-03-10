# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:58:35 2020

@author: sidra

"""
import pandas as pd

voice = pd.read_csv('voice.csv')
print(voice.head())
voice["label"].value_counts()
voice.info()

import seaborn as sns
from seaborn import plt

for col in voice.columns[:-1]:
    sns.FacetGrid(voice, hue="label", size=3).map(sns.kdeplot, col).add_legend()
    plt.show()
sns.FacetGrid(voice, hue="label", size=7).map(plt.scatter, "IQR", "meanfun").add_legend()
plt.show()
sns.FacetGrid(voice, hue="label", size=7).map(plt.scatter, "meanfreq", "meanfun").add_legend()
plt.show()
from sklearn.preprocessing import LabelEncoder

# replace male/female => 1/0
gender_encoder = LabelEncoder()
voice['label'] = gender_encoder.fit_transform(voice.iloc[:, -1])

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

y=voice['label']
x = voice.iloc[:, :-1]
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# code without scale below

#train, test = train_test_split(voice, test_size=0.2, random_state=2)

# separating data in features and labels
#x_train = train.iloc[:, :-1]
#y_train = train.iloc[:, -1]
#x_test = test.iloc[:, :-1]
#y_test = test.iloc[:, -1]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(x_train, y_train)
prediction = lr.predict(x_test)

print('LR result: ', accuracy_score(y_test, prediction))
from sklearn.cluster import KMeans
import math

result = 0
for num_clusters in range(1, len(x_train)):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(x_train, y_train)
    prediction = kmeans.predict(x_test)
    cur_result = accuracy_score(y_test, prediction)
    if cur_result < result or math.fabs(cur_result-result) < 0.01:
        break
    result = cur_result 

print('K-means result: ', result, '. Cluster quantity = ', num_clusters-1)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
prediction = nb.predict(x_test)

print('NB result: ', accuracy_score(y_test, prediction))

from sklearn.svm import SVC

svc=SVC()
svc.fit(x_train,y_train)
prediction = svc.predict(x_test)
print('SVM result: ', accuracy_score(y_test,prediction))

voice_cut = voice[['IQR', 'meanfun', 'label']]

#y_cut=voice['label']
#x = voice_cut.iloc[:, :-1]
#scaler = StandardScaler()
#scaler.fit(x)
#x_cut = scaler.transform(x)

#x_train, x_test, y_train, y_test = train_test_split(x_cut, y_cut, test_size=0.2, random_state=2)

train, test = train_test_split(voice_cut, test_size=0.2, random_state=2)

# separating data in features and labels
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
x_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]


lr = LogisticRegression()
lr.fit(x_train, y_train)
prediction = lr.predict(x_test)

print('LR result with 2 features: ', accuracy_score(y_test, prediction))

result = 0
for num_clusters in range(1, len(x_train)):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(x_train, y_train)
    prediction = kmeans.predict(x_test)
    cur_result = accuracy_score(y_test, prediction)
    if cur_result < result or math.fabs(cur_result-result) < 0.01:
        break
    result = cur_result 

print('K-means result with 2 features: ', result, '. Cluster quantity = ', num_clusters-1)

nb = GaussianNB()
nb.fit(x_train, y_train)
prediction = nb.predict(x_test)

print('NB result with 2 features: ', accuracy_score(y_test, prediction))


svc=SVC()
svc.fit(x_train,y_train)
prediction = svc.predict(x_test)

print('SVM result with 2 features: ', accuracy_score(y_test,prediction))