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

# Feature Reduction

In this notebook we'll take a look at the features in our training set and try to determine if there are any features we can remove.

```python
# Import packages
import sys
sys.path.append('../../')

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import collegebasketball as cbb
cbb.__version__
```

### Read in the Training Data

To start with, we'll read in the training data and take a quick look at the features. We can see below that there are 263 columns in the data set, and though six of them are not feature, we'll try and determine if all of them are needed in our model. 

First, we'll read in the full training data set and we'll remove some features that I know are almost identical to other columns in the data. For example, the `AdjOE` feature from the T-Rank data is essentially the same as the `AdjO` feature from the Kenpom data. Generally, we'll keep the Kenpom version of any features that are common across other data sources.

Next, we'll further split the data into a training and test set so we can verify that the features we removed don't have too large of an impact on the predictive power of our training data.

```python
# Paths to datasets
path_dir = '../../Data/Training/large_training.csv'

# Read in data
df = pd.read_csv(path_dir)
df.columns
```

```python
# We want to drop columns that are almost exactly the same before this analysis
repeat_cols = ['Barthag', 'AdjOE', 'AdjDE', 'Adj T.', 'WAB', 'MP', 'FG%', 'FG%_opp', '3P%', '3P%_opp']

# For this analysis we will not include non-feature, difference or rank columns
cols_to_drop = (
    ['Favored', 'Underdog', 'Year', 'Tournament', 'Seed_Fav', 'Seed'] + 
    [col for col in df.columns if '_Diff' in col or 'Rank' in col or 'Rk' in col] +
    repeat_cols + [col + '_Fav' for col in repeat_cols]
)

tourney = cbb.filter_tournament(df).drop(cols_to_drop, axis=1)
regular = cbb.filter_tournament(df, drop=True).drop(cols_to_drop, axis=1)

print(f'We have {len(regular)} regular season games and {len(tourney)} NCAA Tournament games.')
```

```python
# Split into train and test sets with an equal proportion of actual tournament games in each
regular_train, regular_test = train_test_split(regular, random_state=77)
tourney_train, tourney_test = train_test_split(tourney, random_state=77)

train = pd.concat([regular_train, tourney_train])
test = pd.concat([regular_test, tourney_test])

print(f'There are {len(train)} games in the training set and {len(test)} games in the test set.')
```

```python
clf = RandomForestClassifier(n_estimators=100, random_state=77)
clf.fit(train.drop('Label', axis=1), train['Label'])
print("Accuracy on test data: {:.2f}".format(clf.score(test.drop('Label', axis=1), test['Label'])))
```

We can see that we get an accuracy of 63% when using all of the features from the training data to train a simple random forest model to predict on the test set. After we remove any additional features, we'll run train another model with the reduced data set and test it again to make sure we don't negatively impact performance.

### Examine Correlated Features

We know that there's some overlap between the features from the different data sources, with some being exact copies and others being highly correlated. We've already removed some obviously similar features already, but next we'll take a closer look at the correlation among the remaining features to try and find more redundancies. We'll take a look at a visualization of the correlation matrix between the features as well as the pairs of features with the highest magnitude of correlation to help determine which features we can remove without losing too much predictive signal. We can also use clustering methods to cluster features into groups to help us understand if groups of features - not just pairs - are all similar.

To help simplify this process, we'll only be looking at the features for team considered the underdog. We'll assume that the correlation between features for the favorite follow a similar pattern - that is, if `EFG%` (effective field goal percentage of the underdog) is highly correlated with `2P%` (2 pointer shooting percentage for the underdog), then we will assume the same correlation applies to those two statistics for the favored team.

```python
# Split features in to features for favorite team and those for the underdog
cols_fav = [col for col in train.columns if '_Fav' in col]
X_fav = train[cols_fav]
X_under = train.drop(cols_fav + ['Label'], axis=1)
y = train['Label']

print(len(X_fav.columns), len(X_under.columns))
```

```python
# Since the plots look very similar, we'll just show the plot for the underdog features
f, ax = plt.subplots(figsize=(15, 10))
corr = X_under.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
```

```python
# Let's take a closer look at feature correlations 
corr_list = X_under.corr().stack()

# Remove correlation of a feature to itself and duplicates
corr_list = corr_list[(corr_list.index.get_level_values(0) != corr_list.index.get_level_values(1)) &
                      (corr_list.index.get_level_values(0) < corr_list.index.get_level_values(1))]

# See top correlation by magnitude
corr_list.apply(abs).sort_values(ascending=False).head(10)
```

```python
# Cluster features based on correlation magnitude
corr_linkage = hierarchy.ward(X_under.corr().apply(abs))
cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')

# Show clusters with more than one feature
cluster_id_to_features = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    feat = X_under.columns[idx]
    cluster_id_to_features[cluster_id].append(feat)
    
for k in cluster_id_to_features:
    if len(cluster_id_to_features[k]) > 1:
        print(cluster_id_to_features[k])
```

### Remove Unecessary Features

From the correlation values, we can see that many of the features are highly correlated. By removing features that are highly correlated with at least one feature we are keeping, we can reduce complexity without losing too much additional information that the could have provided. There are a few patterns in the correlation of features above:

1. There are some cases where we have one feature to describe the offense and another feature that describes essentially the same thing from the opposing team's perspective. For example, the `FT` feature and `PF_opp` feature are really both representative of how often the underdog gets fouled. Sure, there are some small differences between the two - you can get fouled without it resulting in free throws - but to help reduce that complexity we'll sacrifice any small bits of information the extra feature could have provided.
2. We often have groups of features from the Sports Reference data set that represent successes, attempts, and success rate. For example, `3P` is the number of made three pointers, `3PA` is the number of threes attempted and `3P%` is the percentage of three pointers made. Since we have the number of attempts and the success rate, the number of successes is redundant and we can remove these features.
3. There are some features from the Sports Reference that are not tempo adjusted that are representative of the same thing as a tempo adjusted version in one of the other two data sets. For exmaple, `TOV` is the number of turnovers, but we have the tempo adjusted version `TOR` from the T-Rank data that is both more accurate and less correlated with the tempo feature.
4. Lastly, there are a couple pairs of features that are very similar to each other that were not removed early. For example, `FGA` and the team's tempo are essentially giving the same information - how many possessions a team gets per game. 

Let's remove the features described above and take another look at some of those correlation metrics from earlier to see how things have changed.

```python
# Remove features where we don't need offensive and defensive numbers 
feats_to_remove = ['FTA_opp', 'PF_opp', 'OppO', 'OppD']

# Remove features that give a success number
feats_to_remove = feats_to_remove + ['FG', 'FG_opp', '3P', '3P_opp', 'FT', 'FT_opp', 'Tm.']

# Remove absolute stats when we have tempo adjusted version
feats_to_remove = feats_to_remove + ['FTA', 'TOV', 'TOV_opp', 'ORB_y', 'ORB_opp', 'TRB', 
                                     'TRB_opp', 'PF', 'STL', 'STL_opp']

# Remove features that are too similar to another
feats_to_remove = feats_to_remove + ['FGA', 'FGA_opp', 'EFG%', 'EFGD%']

X_under_reduced = X_under.drop(feats_to_remove, axis=1)
X_under_reduced.columns
```

```python
f, ax = plt.subplots(figsize=(15, 10))
corr = X_under_reduced.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
```

```python
feats_to_remove_all = feats_to_remove + ['ORB_Fav_y'] + [col + '_Fav' for col in feats_to_remove]
feats_to_remove_all.remove('ORB_y_Fav')
train_reduced = train.drop(feats_to_remove_all, axis=1, errors='ignore')
test_reduced = test.drop(feats_to_remove_all, axis=1, errors='ignore')

X_train_reduced = train_reduced.drop('Label', axis=1)
y_train_reduced = train_reduced['Label']
X_test_reduced = test_reduced.drop('Label', axis=1)
y_test_reduced = test_reduced['Label']

len(train_reduced.columns)
```

```python
corr_list_reduced = X_train_reduced.corr().stack()

# Remove correlation of a feature to itself and duplicates
corr_list_reduced = corr_list_reduced[
    (corr_list_reduced.index.get_level_values(0) != corr_list_reduced.index.get_level_values(1)) &
    (corr_list_reduced.index.get_level_values(0) < corr_list_reduced.index.get_level_values(1))
]

# See top correlation by magnitude
corr_list_reduced.apply(abs).sort_values(ascending=False).head(10)
```

```python
# Cluster features based on correlation magnitude
corr_linkage = hierarchy.ward(X_train_reduced.corr().apply(abs))
cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')

# Show clusters with more than one feature
cluster_id_to_features = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    feat = X_train_reduced.columns[idx]
    cluster_id_to_features[cluster_id].append(feat)
    
for k in cluster_id_to_features:
    if len(cluster_id_to_features[k]) > 1:
        print(cluster_id_to_features[k])
```

### Test how a Model Performs with the Reduced Feature Set

From the correlation matrix, we can clearly see that the number of correlated features has been reduced dramatically. Hopefully we'll be able to create a simpler model that still has relatively the same predictive power with this reduced set of features. 

You might notice that there are still a couple of features in the data that are pretty highly correlated that I decided to keep in the set of features. Notably, I kept the `AdjEM` feature (overall team efficiency) in addition to both the defensive and offensive efficiency numbers. I felt that these features were all too important to exclude any of them, though you could definitely make an argument that the overall efficiency can easily be derived from the other two features. Another set of features that were pretty closely linked were team efficiency stats and features related to strength of schedule. Overall, better teams tend to play harder schedules so it makes sense, but I felt that these features still provided enough of a different look at the teams so I decided to keep them.

Next, we'll test out the reduced feature data with a random forest classifier to help confirm that we haven't removed too much important information that the model needed. Below, we can see that we get the same 63% accuracy on our reduced test set. While we didn't increase accuracy, I still think the reduced feature training data will help us build a more robust model for making NCAA Tournament predictions later on.

```python
clf = RandomForestClassifier(n_estimators=100, random_state=77)
clf.fit(X_train_reduced, y_train_reduced)
print("Accuracy on test data: {:.2f}".format(clf.score(X_test_reduced, y_test_reduced)))
```

```python
# Save all the training data to a csv - including identifier columns
training_data = pd.concat([train_reduced , test_reduced], ignore_index=False)
training_data = df[['Favored', 'Underdog', 'Year', 'Tournament']].join(training_data)
training_data.to_csv('../../Data/Training/training_feat_reduced.csv', index=False)
training_data.head()
```
