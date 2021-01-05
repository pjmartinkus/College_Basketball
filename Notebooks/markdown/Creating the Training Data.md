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
kenpom_season = pd.read_csv('{}Kenpom.csv'.format(load_path))
kenpom_march = pd.read_csv('{}Kenpom_march.csv'.format(load_path))
TRank_season = pd.read_csv('{}TRank.csv'.format(load_path))
TRank_march = pd.read_csv('{}TRank_march.csv'.format(load_path))
stats_season = pd.read_csv('{}Basic.csv'.format(load_path))
stats_march = pd.read_csv('{}Basic_march.csv'.format(load_path))
```

```python
# Generate features for Kenpom data
kenpom_season_vecs = cbb.gen_kenpom_features(kenpom_season)
kenpom_march_vecs = cbb.gen_kenpom_features(kenpom_march)

# Take a look
print("There are {} games in the Kenpom dataset.".format(len(kenpom_season_vecs)))
print("There are {} games in the march Kenpom dataset.".format(len(kenpom_march_vecs)))
kenpom_season_vecs.head(3)
```

Note that there was a mis-match in the column names for the T-Rank data as opposed to the other datasets. T-Rank calls the columns for wins and losses by the shorter 'W' and 'L'. In order to merge with the other datasets properly, we need to rename these columns to match the Kenpom data.

```python
# Fix some feature names
TRank_season.rename(columns={'W_Home': 'Wins_Home', 'W_Away': 'Wins_Away', 
                            'L_Away': 'Losses_Away', 'L_Home': 'Losses_Home'}, inplace=True)
TRank_march.rename(columns={'W_Home': 'Wins_Home', 'W_Away': 'Wins_Away', 
                            'L_Away': 'Losses_Away', 'L_Home': 'Losses_Home'}, inplace=True)

# Generate features for T-Rank data
TRank_season_vecs = cbb.gen_TRank_features(TRank_season)
TRank_march_vecs = cbb.gen_TRank_features(TRank_march)

# Take a look
print("There are {} games in the T-Rank dataset.".format(len(TRank_season_vecs)))
print("There are {} games in the march T-Rank dataset.".format(len(TRank_march_vecs)))
TRank_season_vecs.head(3)
```

```python
# Generate features for basic stats data
stats_season_vecs = cbb.gen_basic_features(stats_season)
stats_march_vecs = cbb.gen_basic_features(stats_march)

# Take a look
print("There are {} games in the basic stats dataset.".format(len(stats_season_vecs)))
print("There are {} games in the march basic stats dataset.".format(len(stats_march_vecs)))
stats_season_vecs.head(3)
```

Now that the features for each dataset have been generated, we can join them all to form one larger set of training data that contains all of their features. Since the basic stats dataset only went back to 2010, this larger set of dataa will be restricted to just the games from 2010 up until now.

Unfortunately, I ran into an issue because the winning percentage data features from the Kenpom and T-Rank datasets appear to be slightly different sometimes. As a temporary fix, I decided to just go with the Kenpom winning percentage for this larger set of data.

```python
# Generate features for each year of data
on_cols_kp_tr = ['Favored', 'Underdog', 'Year', 'Seed_Fav', 'Seed', 'Label', 'AdjEM_Fav', 'AdjEM', 'AdjEM_Diff']
on_cols_stats = on_cols_kp_tr + ['Win_Loss_Fav', 'Win_Loss', 'Win_Loss_Diff']
```

```python
# Add an id column to the kenpom dataset
all_season = kenpom_season_vecs[kenpom_season_vecs['Year'] > 2009]
all_season.reset_index(level=0, inplace=True)

# Create a set of training data for years with all features
all_season = all_season.merge(TRank_season_vecs[TRank_season_vecs['Year'] > 2009], on=on_cols_kp_tr)
all_season = all_season.rename(columns={'Win_Loss_Fav_x': 'Win_Loss_Fav', 'Win_Loss_x': 'Win_Loss', 'Win_Loss_Diff_x': 'Win_Loss_Diff'})
all_season = all_season.drop(['Win_Loss_Fav_y', 'Win_Loss_y', 'Win_Loss_Diff_y'], axis=1)
all_season = all_season.merge(stats_season_vecs, on=on_cols_stats)
all_season = all_season.drop_duplicates('index').drop('index', axis=1)

all_march = kenpom_march_vecs[kenpom_march_vecs['Year'] > 2009].merge(TRank_march_vecs[TRank_march_vecs['Year'] > 2009], on=on_cols_kp_tr)
all_march = all_march.rename(columns={'Win_Loss_Fav_x': 'Win_Loss_Fav', 'Win_Loss_x': 'Win_Loss', 'Win_Loss_Diff_x': 'Win_Loss_Diff'})
all_march = all_march.drop(['Win_Loss_Fav_y', 'Win_Loss_y', 'Win_Loss_Diff_y'], axis=1)
all_march = all_march.merge(stats_march_vecs, on=on_cols_stats)

# Take a look
print("There are {} games in the dataset.".format(len(all_season)))
print("There are {} games in the march dataset.".format(len(all_march)))
all_season.head()
```

## Blocking

We now have features for every game played in division one from the 2002 season to the 2017 season. However, we can improve the accuracy of our models if we remove results unrealated to our test set. Since the goal of this project is to predict specifically games for the NCAA Tournament, we will remove any games with teams that are not good enough.

```python
print('We have Kenpom data for ' + str(len(kenpom_season_vecs) + len(kenpom_march_vecs)) + ' games.')
print(str(len(kenpom_season_vecs[kenpom_season_vecs['Label'] == 1]) 
          + len(kenpom_march_vecs[kenpom_march_vecs['Label'] == 1])) + ' of those games are upsets')
print('We have T-Rank data for ' + str(len(TRank_season_vecs) + len(TRank_march_vecs)) + ' games.')
print(str(len(TRank_season_vecs[TRank_season_vecs['Label'] == 1]) 
          + len(TRank_march_vecs[TRank_march_vecs['Label'] == 1])) + ' of those games are upsets')
print('We have Stats data for ' + str(len(stats_season_vecs) + len(stats_march_vecs)) + ' games.')
print(str(len(stats_season_vecs[stats_season_vecs['Label'] == 1]) 
          + len(stats_march_vecs[stats_march_vecs['Label'] == 1])) + ' of those games are upsets')
```

```python
# Block the feature vector tables for full season data
kenpom_season_vecs = cbb.block_table(kenpom_season_vecs)
TRank_season_vecs = cbb.block_table(TRank_season_vecs)
stats_season_vecs = cbb.block_table(stats_season_vecs)
all_season = cbb.block_table(all_season)
```

```python
# Drop the kenpom columns from the TRank and basic stats data sets now that blocking is completed
drop_cols = ['AdjEM_Fav', 'AdjEM', 'AdjEM_Diff']
TRank_season_vecs = TRank_season_vecs.drop(drop_cols, axis=1)
TRank_march_vecs = TRank_march_vecs.drop(drop_cols, axis=1)
stats_season_vecs = stats_season_vecs.drop(drop_cols, axis=1)
stats_march_vecs = stats_march_vecs.drop(drop_cols, axis=1)
```

```python
print('We have Kenpom data for ' + str(len(kenpom_season_vecs) + len(kenpom_march_vecs)) + ' games.')
print(str(len(kenpom_season_vecs[kenpom_season_vecs['Label'] == 1]) 
          + len(kenpom_march_vecs[kenpom_march_vecs['Label'] == 1])) + ' of those games are upsets')
print('We have T-Rank data for ' + str(len(TRank_season_vecs) + len(TRank_march_vecs)) + ' games.')
print(str(len(TRank_season_vecs[TRank_season_vecs['Label'] == 1]) 
          + len(TRank_march_vecs[TRank_march_vecs['Label'] == 1])) + ' of those games are upsets')
print('We have Stats data for ' + str(len(stats_season_vecs) + len(stats_march_vecs)) + ' games.')
print(str(len(stats_season_vecs[stats_season_vecs['Label'] == 1]) 
          + len(stats_march_vecs[stats_march_vecs['Label'] == 1])) + ' of those games are upsets')
```

```python
# Now save all of the training datasets 
path = '../Data/Training/'
kenpom_season_vecs.to_csv('{}kenpom_season.csv'.format(path), index=False)
kenpom_march_vecs.to_csv('{}kenpom_march.csv'.format(path), index=False)
TRank_season_vecs.to_csv('{}TRank_season.csv'.format(path), index=False)
TRank_march_vecs.to_csv('{}TRank_march.csv'.format(path), index=False)
stats_season_vecs.to_csv('{}stats_season.csv'.format(path), index=False)
stats_march_vecs.to_csv('{}stats_march.csv'.format(path), index=False)
all_season.to_csv('{}all_season.csv'.format(path), index=False)
all_march.to_csv('{}all_march.csv'.format(path), index=False)
```
