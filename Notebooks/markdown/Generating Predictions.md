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

# Generating Predictions

Using the Logistic Regression model that we chose in the Selecting a Model notebook, we will create predictions for the 2021 NCAA Tournament.

```python
# Import packages
import sys
sys.path.append('../')

import pandas as pd
from sklearn.linear_model import LogisticRegression
import collegebasketball as cbb

import warnings
warnings.filterwarnings('ignore')

cbb.__version__
```

## Train the Model

Using the same method as before, we will train the model. To understand how I arrived at this model, please look at the Selecting a Model notebook for more information.

However, there is one major difference in how we will train the model this time. Before, we split the data into training and testing sets, but since we are predicting for new games, we will use all of the training data to train the model.

```python
# Load the csv files that contain the scores/kenpom data
year = 2024
path = f'../Data/Training/training_{year}.csv'
train = pd.read_csv(path)

# Get a sense for the size of each data set
print('Length of training data: {}'.format(len(train)))
```

```python
train.head(3)
```

```python
# Get feature names
exclude = ['Favored', 'Underdog', 'Year', 'Tournament', 'Label']

features = list(train.columns)
for col in exclude:
    features.remove(col)
```

```python
# Train the classifier
log = LogisticRegression(penalty='l2', C=10, solver='liblinear', random_state=77)
log.fit(train[features], train[['Label']])
```

## Get Input Data for this Year

Next, we'll need to get the input data for this year so we can use it to predict game results for tournament games. We'll retrieve data from each source for this year, clean the data and combine it into a single data set.

```python
stats_path = '../Data/SportsReference/' + str(year) + '_stats.csv'
stats = pd.read_csv(stats_path)
stats = cbb.update_basic(stats.rename(index=str, columns={'School': 'Team'}))

# Fix absolute stats to be per game
cols_to_fix = ['3PA', '3PA_opp',  'AST', 'AST_opp', 'BLK', 'BLK_opp']
for c in cols_to_fix:
    stats[c] = stats[c] / stats['G']

stats[stats['Team'] == 'Marquette']
```

```python
kp_path = '../Data/Kenpom/' + str(year) + '_kenpom.csv'
kenpom = pd.read_csv(kp_path)
kenpom = cbb.update_kenpom(kenpom)
kenpom[kenpom['Team'] == 'Marquette']
```

```python
TRank_path = '../Data/TRank/' + str(year) + '_TRank.csv'
TRank = pd.read_csv(TRank_path)
TRank = cbb.update_TRank(TRank)
TRank[TRank['Team'] == 'Marquette']
```

```python
# Merge the data from each source (and drop columns that are repeats)
team_stats = pd.merge(kenpom, TRank.drop(['Conf', 'Wins', 'Losses'], axis=1), on='Team', sort=False)
team_stats = pd.merge(team_stats, stats.drop(['G', 'ORB', '3P%', 'ORB'], axis=1), on='Team', sort=False)
team_stats[team_stats['Team'] == 'Marquette']
```

```python
# Load Tournament games
games_path = '../Data/Tourney/{}.csv'.format(year)
games = pd.read_csv(games_path)
games.head(3)
```

```python
# Join the team data with the game data
data = pd.merge(games, team_stats, left_on='Home', right_on='Team', sort=False, how='left')
data = pd.merge(data, team_stats, left_on='Away', right_on='Team', suffixes=('_Home', '_Away'), sort=False, how='left')
data.insert(0, 'Year', year)
data.insert(3, 'Tournament', 'NCAA Tournament')

# Confirm school names are correct
assert len(data[(data['Rank_Home'].isna()) | (data['Rank_Away'].isna())]) == 0

data.head(3)
```

## Predict Games Using the Classifier

Now that we have a trained model and data for the tournament games this year, we can use it to predict games in the 2021 NCAA Tournament.

```python
# Make Predictions
predictions = cbb.predict(log, data, features)
predictions.to_csv('../Data/predictions/predictions_2024.csv', index=False)
predictions['Upset'] = predictions['Underdog'] == predictions['Predicted Winner']
```

```python
# First Round
predictions.iloc[0:32,:]
```

```python
# Second Round
predictions.iloc[32:48,:]
```

```python
# Later Rounds
predictions.iloc[48:,:]
```

Congratulations to all UConn fans because the model has predicted the Huskies to repeat as the the 2024 NCAA Tournament Champion!

```python

```
