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

## Creating the Training Data

In this notebook, we will create the training data to be used by the various models for predicting scores in the NCAA basketball tournament. We will start by generating features for each dataset. Then, we will use blocking to reduce the training data to only include games that include games between tournament caliber teams. Finally, we will combine data sets to create training data for a machine learning model.

```python
# Import packages
import sys
sys.path.append('../../College_Basketball')

import pandas as pd
import collegebasketball as cbb
cbb.__version__
```

## Feature Generation

Now that we have our data, we need to create some features for the ML algorithms. For each statistical attribute, there is a feature to show the attribute for the favored team, the attribute for the underdog, and the difference between the two. The favored team is defined as the team with a higher AdjEM on kenpom for each dataset. Using this system, a label of '1' represents an upset and a label of '0' means that the favored team won the game.

We will create a dataset with these features for each set of statistics (Kenpom, T-Rank, basic) for each year that these stats are available. Additionally, we will create a dataset that includes all three of these sets of statistics in a single data set.

```python
# Load the joined datasets
load_path = '../Data/Combined_Data/'
kenpom = pd.read_csv(f'{load_path}Kenpom.csv')
TRank = pd.read_csv(f'{load_path}TRank.csv')
stats = pd.read_csv(f'{load_path}Basic.csv')
```

```python
# Generate features for Kenpom data
kenpom_vecs = cbb.gen_kenpom_features(kenpom)

# Take a look
print(f'There are {len(kenpom_vecs)} games in the Kenpom dataset.')
print(f'There are {len(cbb.filter_tournament(kenpom_vecs))} tournament games in the Kenpom dataset.')
kenpom_vecs.head(3)
```

```python
# Generate features for T-Rank data
TRank_vecs = cbb.gen_TRank_features(TRank)

# Take a look
print(f'There are {len(TRank_vecs)} games in the T-Rank dataset.'.format())
print(f'There are {len(cbb.filter_tournament(TRank_vecs))} games in the march T-Rank dataset.')
TRank_vecs.head(3)
```

```python
# Generate features for basic stats data
stats_vecs = cbb.gen_basic_features(stats)

# Take a look
print(f'There are {len(stats_vecs)} games in the basic stats dataset.')
print(f'There are {len(cbb.filter_tournament(stats_vecs))} games in the march basic stats dataset.')
stats_vecs.head(3)
```

Now that the features for each dataset have been generated, we can join them all to form one larger set of training data that contains all of their features. Since the basic stats dataset only went back to 2010, this larger set of data will be restricted to just the games from 2010 up until now.

Unfortunately, I ran into an issue because the winning percentage data features from the Kenpom and T-Rank datasets appear to be slightly different sometimes. As a temporary fix, I decided to just go with the Kenpom winning percentage for this larger set of data.

```python
# Generate features for each year of data
on_cols_kp_tr = ['Favored', 'Underdog', 'Year', 'Tournament', 'Seed_Fav', 'Seed', 'Label', 'AdjEM_Fav', 'AdjEM', 'AdjEM_Diff']
on_cols_stats = on_cols_kp_tr + ['Win_Loss_Fav', 'Win_Loss', 'Win_Loss_Diff']
```

```python
# Add an id column to the kenpom dataset
all_vecs = kenpom_vecs[kenpom_vecs['Year'] > 2009]
all_vecs.reset_index(level=0, inplace=True)

# Create a set of training data for years with all features
all_vecs = all_vecs.merge(TRank_vecs[TRank_vecs['Year'] > 2009], on=on_cols_kp_tr)
all_vecs = all_vecs.rename(columns={'Win_Loss_Fav_x': 'Win_Loss_Fav', 'Win_Loss_x': 'Win_Loss', 
                                    'Win_Loss_Diff_x': 'Win_Loss_Diff'})
all_vecs = all_vecs.drop(['Win_Loss_Fav_y', 'Win_Loss_y', 'Win_Loss_Diff_y'], axis=1)
all_vecs = all_vecs.merge(stats_vecs.drop(['ORB', 'ORB_Fav'], axis=1), on=on_cols_stats)
all_vecs = all_vecs.drop_duplicates('index').drop('index', axis=1)

# Take a look
print("There are {} games in the dataset.".format(len(all_vecs)))
print("There are {} games in the march dataset.".format(len(cbb.filter_tournament(all_vecs))))
all_vecs.head()
```

### Filter Training Set

A problem with our training set here is that many of the games we have data for don't provide useful information about how game in the NCAA Tournament tend to play out. Games between teams that aren't even close to tournament teams are not very predictive of tournament results and therefore we can filter these games out from our training set. 

Again, we previously ran an analysis about how to best filter the training set down to both more closely match actual tournament games without reducing the number of games in the training set too severely. Check out the `covariate_shift.ipynb` notebook for more detailed information.

```python
# First keep all tournament games
t_vecs = cbb.filter_tournament(all_vecs)
r_vecs = cbb.filter_tournament(all_vecs, drop=True)

print("There are {} games in the march dataset.".format(len(t_vecs)))
```

```python
# Our filtering rule is that each of the features below must be at least in the 
# 2nd percentile of the tournament data
threshold_features = ['Win_Loss', 'AdjEM', 'Barthag']
threshold_quatile = 0.02
```

```python
# Filter regular season data
thresholds = t_vecs.quantile(threshold_quatile)[threshold_features]
is_above_threshold = pd.concat([r_vecs[feat] >= thresh for feat, thresh in thresholds.iteritems()], axis=1)
filtered_r_vecs = r_vecs[is_above_threshold.all(axis=1)]

print("There are {} games in the filtered regular season data".format(len(filtered_r_vecs)))
filtered_r_vecs.head(3)
```

```python
# Combine all games in training data
filtered_vecs = pd.concat([filtered_r_vecs, t_vecs])
print("There are {} games in the training data".format(len(filtered_vecs)))
```

### Feature Reduction

Now that we've joined the data from each data source to a single set of features for each game, we've got a dataset with over 250 columns. At least for the models we'll be looking at, having so many features - many of which are correlated with each other - can be a problem and lead to a less robust model. To help reduce the correlation among the features in our training data, we will choose just the most important features with the least amount of overlap in gained information among them.

Luckily, we've already done an analysis on which features should be in the feature set in the `feature_reduction.ipynb` over in the analysis directory. For more detailed information on why these features were chosen, check out that notebook.

```python
feature_names = ['Win_Loss', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'OppAdjEM',
               'NCSOS AdjEM', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%',
               '2P%D', '3P%D', '3PA', '3PA_opp', 'FT%', 'FT%_opp', 'AST', 'AST_opp',
               'BLK', 'BLK_opp']

feature_set = list()
for n in feature_names:
    feature_set.append(n + '_Fav')
    feature_set.append(n)

needed_cols = ['Favored', 'Underdog', 'Year', 'Tournament', 'Label']
```

```python
feature_vecs = filtered_vecs[needed_cols + feature_set]
feature_vecs.head(3)
```

```python
# Save final training set
year = 2024
feature_vecs.to_csv(f'../Data/Training/training_{year}.csv', index=False)
```

```python

```
