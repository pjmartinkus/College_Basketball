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

First, we need to load the training data. This data set was built by creating feature vectors, removing games that don't accurately represent the NCAA Tournament games and reducing the number of features to a manageable number. More information about how this was done can be found in the Creating the Training Data, Covariate Shift Analysis and Feature Reduction notebooks. We'll split this data into training and test sets (with an even proportion of regular season and tournament games in each) so that we can tune the models with the training set before finally testing them with the test set.

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

To start with, we'll try to tune some parameters for a logistic regression model. We'll try both Ridge and Lasso regression with different values for the `C` parameter (where a smaller value imposes a harsher penalty). To compare the different models, we'll run five fold cross validation on the test set and compare the average f1 and AUC scores across the different runs.

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

From the cross validation results above, I think both types of regularization penalties work well with a C value of 10. Since the chance of an upset is extremely variable, we want a robust model with a harsh enough regularization penalty, so I don't want the C value to be too high, but the f1 score does appear to increase as we increase C. I think 10 is a good middle ground and what we will use moving forward.

Below, we run train a model with each set of parameters on regular season data from the training data and evaluate them with the tournament data from the training set, just to see how well they seem to be able to predict tournament games. We can see that the performance is very similar between the two models, but there is a slight advantage for the Ridge Regression model.

In the end, the best logistic regression parameters for me are to use l2 regularization with a C value of 10. I think this will be the most robust logistic regression model we've tested here and we will compare it to the other algorithms later in the notebook.

```python
clf_l2 = LogisticRegression(penalty='l2', C=10, solver='liblinear', random_state=77)
clf_l1 = LogisticRegression(penalty='l1', C=10, solver='liblinear', random_state=77)
cbb.evaluate(season_train, march_train, exclude_cols, [clf_l2, clf_l1], ['L2 Penalty', 'L1 Penalty'])
```

## Random Forest Tuning

Next, we will follow a similar process to tune a random forest model. This time, we will be trying different values for the minimum number of samples to split, the maximum depth of trees and the maximum features used per tree. For reference, the default values in scikit-learn are 2, None and 'sqrt' respectively.

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

While there are a lot of results to process above, I found that some patterns emerged for each of the parameters we are tuning:
* min_samples_split: There is the least amount of consensus for this parameter, It seems like all of the options I tried worked well with at least some combination of the others. However, since 2 is the default, I will generally favor using that value.
* max_depth: It seems that generally the models with a larger maximum depth performance better. However, smaller and therefore simpler trees should be more robust, so I would like to use 25 as a maximum depth if possible.
* max_features: Generally, the models with more features performed better. Since there are a good number of features in this data set and we have already worked to try and reduce that number as much as possible, I think it is alright to use up to ten features for the trees.

Based on the trends above, I picked out some of the combinations of parameters to test them out on the training tournament data. Overall, I settled on using the default 2 for min_samples_split, a max depth of 25 and a max number of features of 10.

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

Lastly, we will also take a look at using an XGBoost model. I have the least amount of familiarity with this type of model, so I'm just going to stick with the default values for now. Regardless, we'll run some cross validation and see how the model performs when trained on the regular season training data and tested on the march training data.

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

Now that we've found the set of ideal parameters for each of the algorithms we'll be testing, we can compare the results of training each on the training data and then testing on the test data.

```python
# Create the models
log = LogisticRegression(penalty='l2', C=10, solver='liblinear', random_state=77)
rf = RandomForestClassifier(n_estimators=500, max_depth=25, max_features=10, random_state=77)
xgb = XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', random_state=77)

clfs = [log, rf, xgb]
cl_names = ['Logistic Regression', 'Random Forest', 'XGBoost']

cbb.evaluate(season, march, exclude_cols, clfs, cl_names)
```

The results above show that the Logistic Regression model is performing best when we look at precision, recall, f1 and AUC scores. Interestingly, the other models both have a higher Brier score, which could indicate that while they're prediction may be less accurate, they are generating better probability values.

Before definitively selecting one of these models, I also want to drill down a little more specifically on how well the models are able to predict NCAA Tournament games. After all, that is the goal of this project. In order to gauge the models' accuracy for those games, I've developed a version of cross validation that I like to call "leave march out" cross validation. The idea is to leave the games from one year's NCAA Tournament aside per fold, train the models on the rest of the data and then test them on that year's set of tournament games. This way, each fold is using most of the data we have for training and testing on a smaller set of games that are all NCAA Tournament games. The output shows how well the model performed on the games from each year's NCAA Tournament.

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

The above results support out finding from using the training and test sets - the logistic regression model is performing best by each metric except the Brier score.

I think it's important to select a model that is able to generate useful prediction probabilities. Since the NCAA Tournament notoriously has so many upsets, I want to be sure that my model will predict enough upsets. To achieve this goal, I've found that in the past, I had to lower the threshold for an upset form the usual probability of 0.5, to something lower and in order for this technique to be effective, we need a model that is generating accurate probability values. 

One method to evaluate these probabilities is to use something like the Brier score used above or another similar method like Log Loss. However, I also wanted to use a more visual method to examine these probabilities closer. To do this, I used a binning method to group sets of games by the predicted probability of an upset assigned by a given model. Then I calculate the actual fraction of upsets in each bin and plotted the results. Ideally, if the model assigned a probability between 0.5 and 0.55 for a set of games, something like 55% of the games should have resulted in an upset. The closer that the resulting plots follow a slope of 1.0, the better it would appear that the model is predicting probabilities.

```python
probs = [log_data['Probability'], rf_data['Probability'], xgb_data['Probability']]
cbb.probability_graph(log_data['Label'], probs, cl_names, figsize=(10, 6))
```

The results above show that the logistic regression and random forest models are producing better probabilities, which is directly at odds with the Brier scores we saw above. The chances a game is an upset seems to stay relatively level regardless of what the XGBoost model actually predicted as the predicted probability.

Overall, I think I will go with the logistic regression model this year because it had the best performance by most metrics, is the simplest and seems to have relatively useful probability scores.
