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

# Generating Predictions

Using the AdaBoost model that we selected in the Selecting a Model notebook, we will create preditions for the 2018 NCAA Tournament.

```python
# Import packages
import sys
sys.path.append('/Users/phil/Documents/Documents/College_Basketball')

import pandas as pd
import collegebasketball as cbb
cbb.__version__

import warnings
warnings.filterwarnings('ignore')
```

## Train the Model

Using the same method as before, we will train the model. To understand how I arrived at this model, please look at the Selecting a Model notebook for more information.

However, there is one major difference in how we will train the model this time. Since we were using the tournament data as a test set before, we did not use it to train the model. However, since we are now predicting on the 2018 data, we can use the tournament data to help train the model.

```python
# Load the csv files that contain the scores/kenpom data
path = '../Data/Training/'
kenpom_season = cbb.load_csv('{}kenpom_season.csv'.format(path))
kenpom_march = cbb.load_csv('{}kenpom_march.csv'.format(path))

# Get a sense for the size of each data set
print('Length of kenpom data: {}'.format(len(kenpom_season) + len(kenpom_march)))
```

```python
# Combine regular season and march data
kenpom_data = pd.concat([kenpom_season, kenpom_march])

# Get feature names
exclude = ['Favored', 'Underdog', 'Year', 'Label']
features = list(kenpom_season.columns)
for col in exclude:
    features.remove(col)
```

```python
# Train the classifier
log = cbb.LogisticRegression(penalty='l1', C=10)
log.fit(kenpom_data[features], kenpom_data[['Label']])
```

## Predict Games Using the Classifier

Now that we have a trained model, we can use it to predict games in the 2018 NCAA Tournament. First we need to load in the games from the first round and create feature vectors.

```python
# Load games csv
games = cbb.load_csv('/Users/phil/Documents/Documents/College_Basketball/Data/Tourney/2019.csv')
games.head()
```

```python
# Set up the training data set
path = '/Users/phil/Documents/Documents/College_Basketball/Data/Kenpom/2019_kenpom.csv'
kenpom = cbb.load_csv(path)
kenpom = cbb.update_kenpom(kenpom)
kenpom.head()
```

```python
# Merge Games data with the different data sets
games = pd.merge(games, kenpom, left_on='Home', right_on='Team', sort=False)
games = pd.merge(games, kenpom, left_on='Away', right_on='Team', suffixes=('_Home', '_Away'), sort=False)
games.insert(0, 'Year', 2019)

games.head()
```

Now that we have feature vectors for the first round of the tournament and a trained model, we can make our predictions for the 2018 NCAA Tournament.

```python
# Make Predictions
predictions = cbb.predict(log, games, features)
```

```python
# First Round
predictions.iloc[0:32,:]
```

```python
seeds = []
for i in range(4):
    seeds.extend([1, 8, 5, 4, 6, 3, 7, 2])
data = predictions[0:32].copy()
data['Top Seed'] = seeds

data = data.sort_values(by=['Top Seed', 'Probabilities'])

winner = []
count = 0
for row in data.iterrows():
    if row[1]['Top Seed'] < 5:
        winner.append(row[1].loc['Favored'])
    elif count < 2:
        winner.append(row[1].loc['Favored'])
    else:
        winner.append(row[1].loc['Underdog'])
    count = count + 1
    if count > 3:
        count = 0
data['Winner'] = winner

data = data.sort_index()
actual_winner = ['Duke', 'UCF', 'Liberty', 'Virginia Tech', 'Maryland', 'LSU',
                'Minnesota', 'Michigan State', 'Gonzaga', 'Baylor', 'Murray State',
                'Florida State', 'Buffalo', 'Texas Tech', 'Florida', 'Michigan',
                'Virginia', 'Oklahoma', 'Oregon', 'UC Irvine', 'Villanova',
                'Purdue', 'Iowa', 'Tennessee', 'UNC', 'Washington', 'Auburn',
                'Kansas', 'Ohio State', 'Houston', 'Wofford', 'Kentucky']
data['Actual Winner'] = actual_winner

print(sum(data['Winner'] == data['Actual Winner']))
print(sum(data['Winner'] == data['Predicted Winner']))

data
```

```python
# Second Round
predictions.iloc[32:48,:]
```

```python
# Later Rounds
predictions.iloc[48:,:]
```

```python

```
