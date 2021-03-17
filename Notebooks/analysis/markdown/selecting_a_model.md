---
jupyter:
  jupytext:
    formats: ipynb,markdown//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
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
sys.path.append('../../')

import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import collegebasketball as cbb
cbb.__version__
```

## Load the Data

First, we need to load the training data. This data set was built by creating feature vectors, removing games that don't accurately represent the NCAA Tournament games and reducing the number of features to a manageable number. More information about how this was done can be found in the Creating the Training Data, Covariate Shift Analysis and Feature Reduction notebooks.

```python
# Load the csv files that contain the scores/kenpom data
path = '../../Data/Training/training_feat_reduced.csv'
training = pd.read_csv(path)
season = cbb.filter_tournament(training, drop=True)
march = cbb.filter_tournament(training)
exclude_cols = ['Favored', 'Underdog', 'Year', 'Tournament', 'Label']

# Get a sense for the size of each data set
print('Games in training data: {}'.format(len(training)))
print('Regular season games: {0}, Tournament games: {1}'.format(len(season), len(march)))
```

```python
# Split into train and test sets with an equal proportion of actual tournament games in each
season_train, season_test = train_test_split(season, random_state=77)
march_train, march_test = train_test_split(march, random_state=77)

train = pd.concat([season_train, march_train])
test = pd.concat([season_test, march_test])

print('Games in training set: {}'.format(len(train)))
print('Games in test set: {}'.format(len(test)))
```

## Logistic Regression Tuning

Test with l1 and l2 regularization and different c values

```python
penalties = ['l1', 'l2']
c_values = [0.1, 1, 10, 50, 100]

cl_names = list()
clfs = list()
for p in penalties:
    for c in c_values:
        cl_names.append('log_reg_{0}_{1}'.format(p, c))
        clfs.append(LogisticRegression(penalty=p, C=c, solver='liblinear', random_state=77))

cbb.cross_val(train, exclude_cols, clfs, cl_names)
```

```python
cbb.cross_val(train, exclude_cols, clfs, cl_names, scoring='roc_auc')
```

```python
clf_l2 = LogisticRegression(penalty='l2', C=10, solver='liblinear', random_state=77)
clf_l1 = LogisticRegression(penalty='l1', C=10, solver='liblinear', random_state=77)
cbb.evaluate(season_train, march_train, exclude_cols, [clf_l2, clf_l1], ['L2 Penalty', 'L1 Penalty'])
```

## Random Forest Tuning

```python
min_samples_splits = [2, 5, 10]
max_depths = [None, 25, 15]
max_features = ['sqrt', 'log2', 10]

cl_names = list()
clfs = list()
for s in min_samples_splits:
    for d in max_depths:
        for f in max_features:
            cl_names.append('rf_{0}_{1}_{2}'.format(s, d, f))
            clfs.append(RandomForestClassifier(min_samples_split=s, max_depth=d, max_features=f, 
                                               n_jobs=-1, random_state=77))

cross_val_result = cbb.cross_val(train, exclude_cols, clfs, cl_names)
cross_val_result
```

```python
cross_val_result = cbb.cross_val(train, exclude_cols, clfs, cl_names, scoring='roc_auc')
cross_val_result
```

```python
clf_2_None_10 = RandomForestClassifier(n_estimators=500, min_samples_split=2, max_depth=None, max_features=10, random_state=77)
clf_2_25_10 = RandomForestClassifier(n_estimators=500, min_samples_split=2, max_depth=25, max_features=10, random_state=77)
clf_5_None_10 = RandomForestClassifier(n_estimators=500, min_samples_split=5, max_depth=None, max_features=10, random_state=77)
clf_10_25_10 = RandomForestClassifier(n_estimators=500, min_samples_split=10, max_depth=25, max_features=10, random_state=77)
clf_5_None_sqrt = RandomForestClassifier(n_estimators=500, min_samples_split=5, max_depth=None, max_features='sqrt', random_state=77)
clf_10_15_sqrt = RandomForestClassifier(n_estimators=500, min_samples_split=10, max_depth=15, max_features='sqrt', random_state=77)

clfs = [clf_2_None_10, clf_2_25_10, clf_5_None_10, clf_10_25_10, clf_5_None_sqrt, clf_10_15_sqrt]
cl_names = ['rf_2_None_10', 'rf_2_25_10', 'rf_5_None_10', 'rf_10_25_10', 'rf_5_None_sqrt', 'rf_10_15_sqrt']
cbb.evaluate(season_train, march_train, exclude_cols, clfs, cl_names)
```

## Taking a Look at an XGBoost Model

```python
clf = XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=77)
cbb.cross_val(train, exclude_cols, [clf], ['XGBoost'])
```

```python
cbb.cross_val(train, exclude_cols, [clf], ['XGBoost'], scoring='roc_auc')
```

```python
cbb.evaluate(season_train, march_train, exclude_cols, [clf], ['XGBoost'])
```

## Selecting a ML Model

Next, we will train the models and test them using the march data to select a ML model that can best predict upsets in the NCAA Tournament for each training data set. We will be choosing between  Decision Tree, Random Forest and Logistic Regression. We will be using classifiers from scikit learn.

```python
# Create the models
log = LogisticRegression(penalty='l2', C=10, solver='liblinear', random_state=77)
rf = RandomForestClassifier(n_estimators=500, max_depth=25, max_features=10, random_state=77)
xgb = XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=77)

clfs = [log, rf, xgb]
cl_names = ['Logistic Regression', 'Random Forest', 'XGBoost']

cbb.evaluate(season, march, exclude_cols, clfs, cl_names)
```

```python
cv_results, log_data = cbb.leave_march_out_cv(season, march, exclude_cols, log)
cv_results
```

```python
cv_results, rf_data = cbb.leave_march_out_cv(season, march, exclude_cols, rf)
cv_results
```

```python
cv_results, xgb_data = cbb.leave_march_out_cv(season, march, exclude_cols, xgb)
cv_results
```

```python
probs = [log_data['Probability'], rf_data['Probability'], xgb_data['Probability']]
probability_graph_new(data_with_predictions['Label'], probs, cl_names)
```

The results above show that the Logistic Regression Model on just the Kenpom works best. For now, we will just use this model with this data set. In the future, I plan to use multiple models on each of these different data sets weighted by thier performance, but for now we will just use this model.

```python

```
