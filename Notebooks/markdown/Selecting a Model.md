---
jupyter:
  jupytext:
    formats: ipynb,markdown//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Selecting a Model

The goal of this notebook is to choose the best model for predicting scores of NCAA Tournament games. We will use the training data to run several machine learning models for our data. Finally, using the results, we select a modle to use for our NCAA Tournament predictions.

```python
# Import packages
import sys
sys.path.append('../')

import datetime
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import collegebasketball as cbb
cbb.__version__
```

## Load the Data

First, we need to load the data that we previously retrieved and cleaned. More information about how this was done can be found in the Data Preparation and Cleaning notebook.

```python
# Load the csv files that contain the scores/kenpom data
path = '../Data/Training/'
kenpom_season = cbb.load_csv('{}kenpom_season.csv'.format(path))
kenpom_march = cbb.load_csv('{}kenpom_march.csv'.format(path))
TRank_season = cbb.load_csv('{}TRank_season.csv'.format(path))
TRank_march = cbb.load_csv('{}TRank_march.csv'.format(path))
stats_season = cbb.load_csv('{}stats_season.csv'.format(path))
stats_march = cbb.load_csv('{}stats_march.csv'.format(path))
all_season = cbb.load_csv('{}all_season.csv'.format(path))
all_march = cbb.load_csv('{}all_march.csv'.format(path))

# Get a sense for the size of each data set
print('Length of kenpom data: {}'.format(len(kenpom_season) + len(kenpom_march)))
print('Length of TRank data: {}'.format(len(TRank_season) + len(TRank_march)))
print('Length of basis stats data: {}'.format(len(stats_season) + len(stats_march)))
print('Length of all data: {}'.format(len(all_season) + len(all_march)))
```

## Selecting a ML Model

Next, we will train the models and test them using the march data to select a ML model that can best predict upsets in the NCAA Tournament for each training data set. We will be choosing between K Nearest Neighbors, Decision Tree, Random Forest, SVM, and Logistic Regression. We will be using classifiers from scikit learn.

```python
# Create the models
knn =  KNeighborsClassifier()
dt = DecisionTreeClassifier(min_samples_leaf=5)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=5)
log = LogisticRegression(penalty='l1', C=10)

cls = [knn, dt, rf, log]
cl_names = ['KNN', 'Decision Tree', 'Random Forest', 'Logistic Regression']
exclude = ['Favored', 'Underdog', 'Year', 'Label']
```

```python
# Kenpom Data
cbb.evaluate(kenpom_season, kenpom_march, exclude, cls, cl_names)
```

```python
# TRank Data
cbb.evaluate(TRank_season, TRank_march, exclude, cls, cl_names)
```

```python
# Basic Stats Data
cbb.evaluate(stats_season, stats_march, exclude, cls, cl_names)
```

```python
# All Data
cbb.evaluate(all_season, all_march, exclude, cls, cl_names)
```

The results above show that the Logistic Regression Model on just the Kenpom works best. For now, we will just use this model with this data set. In the future, I plan to use multiple models on each of these different data sets weighted by thier performance, but for now we will just use this model.
