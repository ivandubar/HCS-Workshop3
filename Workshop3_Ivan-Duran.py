#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 00:56:48 2020

@author: ivanduran
"""

from sklearn import datasets
import pandas as pd
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

iris = datasets.load_iris()
iris

# print labels 
print(iris.target_names)

# print features
print(iris.feature_names)

# print the iris labels (0:setosa, 1:versicolor, 2:virginica)
print(iris.target)

df = pd.DataFrame({
    'sepal_length':iris.data[:,0],
    'sepal_width':iris.data[:,1],
    'petal_length':iris.data[:,2],
    'petal_width':iris.data[:,3],
    'species':iris.target
})

# features
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# labels
y = df['species'] 

# create random forest classifier
model = RandomForestClassifier(n_estimators=500, random_state =1)

# fit model w/ 10-fold CV, use StratifiedKFold CrossVal
scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"]
scores = cross_validate(model, X, y, cv=9, scoring=scoring)

print("Average Accuracy (w/ 95% Confidence Interval):")
for metric in scores.keys():
    se = scipy.stats.sem(scores[metric])
    ci = se * scipy.stats.t.ppf((1 + 0.95) / 2., len(scores[metric]) - 1)
    print(f"{metric}: %0.2f (+/- %0.2f)" % (scores[metric].mean()*100, ci*100))
print("Accuracy for each fold: ", scores["test_accuracy"])