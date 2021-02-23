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

 ## Covariate Shift Analysis

In this notebook, we will take a look at our training data and try to figure out how we might be able to create a set of data that accurately represents the NCAA Tournament games we are trying to predict.

Up to this point, we've collected college basketball data from the last ten to fifteen years which includes individual game scores as well as team statistics over each season. The hope is to use these games as a basis to understand patterns in college basketball games to help us predict winners in NCAA Tournament games. However, we first need to find out if this set of data, which includes both regular season and past tournament games, is actually representative of the games we are trying to predict.

This is where [covariate shift analysis](https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/) can come in handy. While this technique is often used to understand if data has changed over time, we can use it to determine if the regular season data we've collected is going to be useful for predicting NCAA Tournament games.

```python
# Import packages
import sys
sys.path.append('../../')

import datetime
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import collegebasketball as cbb
cbb.__version__
```

### Load in the Feature Vectors

To start with we'll load in all of our feature vectors that were generated using other notebooks. Each feature vector represents a single game of college basketball. Features with the suffix `_Fav` indicate a season statistic for the favored team in the given game and other features are for a season statistic for the underdog in that game.

For this analysis, we'll really just be focussing on the data set of feature vectors from each of our input data sources.

```python
# Paths to datasets
path_dir = '../../Data/Feature_Vectors/'
```

```python
# Load in data
kenpom = pd.read_csv(path_dir + 'kenpom.csv')
tRank = pd.read_csv(path_dir + 'TRank.csv')
stats = pd.read_csv(path_dir + 'stats.csv').drop(['Opp._Fav', 'Opp.', 'Opp._Diff'], axis=1)
all_data = pd.read_csv(path_dir + 'training.csv')
all_data = all_data.assign(Seed_Diff=all_data['Seed_Diff_x']) \
    .drop(['Seed_Diff_x', 'Seed_Diff_y', 'Opp._Fav', 'Opp.', 'Opp._Diff'], axis=1)

# Take a look at the data
kenpom.head(3)
```

### Covariate Shift Analysis for Regular Season and NCAA Tournament Data

Now, we'll use the `covariate_shift()` function to help us understand how different our two data sets are. The function will mark each record as either a regular season or tournament game and then will train a random forest classifier to try and predict which data set a given game is from. If the classifier is able to correctly classify a test set of game data, then we know the regular season and tournament data sets are pretty different.

```python
# Function to run covariate shift multiple times and take average for each metric
def cov_shift(train, test, repeat=5):
    metrics = list()
    for _ in range(repeat):
        metrics.append(cbb.covariate_shift(train, test))
    auc = np.mean([x[0] for x in metrics])
    f1 = np.mean([x[1] for x in metrics])
    return auc, f1
```

```python
# Run the covariate shift function on the data
data = [kenpom, tRank, stats, all_data]
names = ['Kenpom', 'T-Rank', 'Basic Stats', 'All Data']

for df, name in zip(data, names):

    test = cbb.filter_tournament(df)
    train = cbb.filter_tournament(df, drop=True)
    mcc, f1 = cov_shift(train, test)
    print('{0}: auc = {1}, f1 = {2}'.format(name, mcc, f1))
```




### Which Features have the Most Shift?

In general we can see that the season and tournament data look very different. The random forest classifier was able to easily predict which games were from the regular season and which were from the tournament. This makes intuitive sense because there are tons of regular season games involving teams that would never be good enough to make the tournament.

Next we'll try and zero in on which features show the biggest difference and we will try and use them later on to filter the regular season data to more like tournament data.

```python
df = all_data.copy()
test = cbb.filter_tournament(df)
train = cbb.filter_tournament(df, drop=True)
```

```python
# Create list of features to test
non_feature_cols = ['Favored', 'Underdog', 'Year', 'Label', 'Tournament',
                    'Seed', 'Seed_Fav', 'Seed_Diff']
feature_to_test = [f for f in df.columns if (f not in non_feature_cols 
                                             and '_Diff' not in f 
                                             and 'Rank' not in f and 'Rk' not in f)]

results = list()
for feature in feature_to_test:
    metrics = list()
    auc, f1 = cov_shift(train[[feature]], test[[feature]])
    results.append((feature, auc, f1))
```

```python
# Save results in a dataframe and sort by MCC
results_df = pd.DataFrame(results, columns=['Feature', 'AUC', 'F1']).sort_values('AUC', ascending=False)
results_df.head(15)
```

## Distribution of Important Features

We can see from the table above which features show the largest difference between the regular season and NCAA Tournament data. Most of them are general stats indicating how good a team is. For example, `Win_Loss` is the winning percentage for the underdog, while `WAB` is wins above bubble stat from T-Rank. Since generally good teams play in the NCAA Tournament games, it makes sense that these types of stats are good indicators of if a game is played between tournament qulaity teams. On the onther hand, more specific team stats are not as useful for this purpose because it is very possible for a team with a weakness, say rebounding or turning the ball over, to still make the tournament due to other strengths. With this in mind, we will focus on some of the features indicated in the table above to help filter non-tournmanet quality games from the training set.

As a next step, we will just take a look at the distributions of some of these features. Each chart below shows the KDE plot of each feature for both the regular season and NCAA Tournament games. This should help us get a better idea of how these features are so useful for differentiating between the two data sets.

```python
def kde_plot(feature, train, test, filtered=[], filtered_names=None):
    if filtered_names is not None:
        names = filtered_names
    else:
        names = [f'Rule {i}' for i, _ in enumerate(filtered)]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    sns.kdeplot(train[feature], ax=ax[0], color = 'darkblue', linewidth=2, label='train')
    sns.kdeplot(test[feature], ax=ax[0], color = 'red', linewidth=2, label='test')
    for i, f in enumerate(filtered):
        sns.kdeplot(f[feature], ax=ax[0], linewidth=2, label=names[i])
    ax[0].legend()

    sns.kdeplot(train[feature + '_Fav'], ax=ax[1], color = 'darkblue', linewidth=2, label='train')
    sns.kdeplot(test[feature + '_Fav'], ax=ax[1], color = 'red', linewidth=2, label='test')
    for i, f in enumerate(filtered):
        sns.kdeplot(f[feature], ax=ax[1], linewidth=2, label=names[i])
    ax[1].legend()
```

```python
kde_plot('Win_Loss', train, test)
```

```python
kde_plot('WAB', train, test)
```

```python
kde_plot('AdjEM', train, test)
```

```python
kde_plot('AdjO', train, test)
```

```python
kde_plot('Barthag', train, test)
```

```python
kde_plot('OppD', train, test)
```

```python
kde_plot('AdjD', train, test)
```

## Looking For Filtering Rules:

In the figures above, we can see that the NCAA Tournament data is more concentrated towards better values, while the regular season data is distributed over a larger range of values. We will next try and use the tournament data to help filter out data for games between teams that would likely not be able to make the NCAA Tournament.

To accomplish this goal, we'll examine different thresholds in the NCAA Tournament data, and then filter out games where that threshold is not met for a given feature. For example, say we are using the `Win_Loss` feature and the quantile $0.05$. We will find the 5$^{th}$ percentile of the `Win_Loss` feature in the NCAA Tournament data and filter out any game in the regular season data that has a `Win_Loss` value below that number. The hope is that by using combinations of features and thresholds, we can remove games from the regular season data that are between non-tournament quality teams and reduce the covariate shift between the two data sets.

```python
# These appear to be the most important features
important_features = [
    'Win_Loss', 'WAB', 'AdjEM', 'Barthag'
]
important_features = important_features + [x + '_Fav' for x in important_features]

# Test all different combinations of features to use for rules
features_to_test = list()
for i in range(1, len(important_features)):
    features_to_test.extend([x for x in combinations(important_features, i)])
    
# Test using different cutoffs for each combination
quantiles =  [q / 100 for q in range(6)]
```

```python
# Function to filter training data using the quantile value of the test data for the given features
def filter_train(features, quantile):
    thresholds = test.quantile(quantile)[features]
    is_above_threshold = pd.concat([train[feat] >= thresh for feat, thresh in thresholds.iteritems()], axis=1)
    return train[is_above_threshold.all(axis=1)]
```

```python
# Start with initial values with no filtering
auc, f1 = cbb.covariate_shift(train, test)
tuning_df = pd.DataFrame({'Features': ['None'], 'Quantile': np.nan, 'auc': auc, 
                          'f1': f1, 'Size': len(train)})


# Test each combination of features/cutoffs
for features in features_to_test:
    for q in quantiles:
        if not isinstance(features, list):
            features = list(features)
        
        filtered = filter_train(features, q)
        auc, f1 = cov_shift(filtered, test)
        tuning_df = tuning_df.append({'Features': features, 'Quantile': q, 'auc': auc, 
                                      'f1': f1, 'Size': len(filtered)},
                                     ignore_index=True)

tuning_df.sort_values('auc').head()
```

```python
tuning_df[tuning_df['auc'] < 0.70].sort_values('Size', ascending=False).head(15)
```

```python
tuning_df[tuning_df['Size'] > 12000].sort_values('auc').head(15)
```

```python
# Save data to csv since it takes some time to run
tuning_df.to_csv('../../Data/analysis/cov_shift.csv', index=False)
```

## Evaluating Rules

In the tables above, we can see different combinations of features, the quantile value used to find thresholds, the average AUC/F1 metrics after running the covariate shift analysis 5 times and the number of games that remained in filtered data set.

After looking through the values, I settled on two different rules: 
1) Using a threshold of the $2^{nd}$ percentile of the `Win_Loss`, `AdjEM`, and `Barthag` features
2) Using a threshold of the $3^{nd}$ percentile of the `Win_Loss`, `WAB`, and `AdjEM_Fav` features

In general, I liked rules that used the `Win_Loss` feature because it was both the most important for the covariate shift analysis and also because it is a pretty basic stat. I also liked to pair it with more advanced stats from different data sets (for example `AdjEM` and `Barthag` are from the Kenpom and T-Rank data sets respectively). 

The first rule I call the "Lenient" rule because it left more games unfiltered in the final data set (14,050 games) and the second rule is the "Strinct" rule because it left only 6,472 games after filtering. I also will include another rule in the comparison below: I filtered out all games that were not between two teams that ended up in the tournament. Hopefully, we can see how each of these data sets compare to the original regular season and NCAA tournament sets as well as to each other.

```python
rules = [
    (['Win_Loss', 'AdjEM', 'Barthag'], 0.02), # Lenient Rule
    (['Win_Loss', 'WAB', 'AdjEM_Fav'], 0.03)  # Strict rule
]
filtered_dfs = [filter_train(f, q) for f, q in rules]

# Also we'll see what filtering by games where both teams were tournament teams
filtered_dfs.append(train[(train['Seed'].notna()) & (train['Seed_Fav'].notna())])

names = ['lenient', 'strict', 'seeded']
```

```python
pd.DataFrame({
    'Name': names,
    'Features': [x[0] for x in rules] + [None],
    'Quantile': [x[1] for x in rules] + [None],
    'Size': [len(df) for df in filtered_dfs],
    'Cov Shift AUC': [cov_shift(df, test)[0] for df in filtered_dfs]
})
```

```python
# Create list of features to test
non_feature_cols = ['Favored', 'Underdog', 'Year', 'Label', 'Tournament',
                    'Seed', 'Seed_Fav', 'Seed_Diff']
feature_to_test = [f for f in df.columns if (f not in non_feature_cols 
                                             and '_Diff' not in f 
                                             and 'Rank' not in f and 'Rk' not in f)]

results = list()
for feature in feature_to_test:
    metrics = list()
    auc_full, f1_full = cov_shift(train[[feature]], test[[feature]])
    auc_lenient, f1_lenient = cov_shift(filtered_dfs[0][[feature]], test[[feature]])
    auc_strict, f1_strict = cov_shift(filtered_dfs[1][[feature]], test[[feature]])
    auc_seed, f1_seed = cov_shift(filtered_dfs[2][[feature]], test[[feature]])
    
    results.append((feature, auc_full, f1_full, auc_lenient, f1_lenient,
                    auc_strict, f1_strict, auc_seed, f1_seed))
    
# Save results in a dataframe and sort by MCC
cols = ['Feature', 'AUC (train)', 'F1 (train)', 'AUC (lenient)', 'F1 (lenient)',
        'AUC (strict)', 'F1 (strict)', 'AUC (seed)', 'F1 (seed)']
results_df = pd.DataFrame(results, columns=cols).sort_values('AUC (train)', ascending=False)
results_df.head(10)
```

The table above shows the results of running the covariate shift analysis for just listed feature values  for each of the various data sets when compared to the NCAA tournament data. We can see how the "strict" and "seeded" data is very close to the tournament data gererally.

Next, we will look feature distributions for each of the original and filtered data sets.

```python
kde_plot('Win_Loss', train, test, filtered_dfs, names)
```

```python
kde_plot('AdjEM', train, test, filtered_dfs, names)
```

```python
kde_plot('WAB', train, test, filtered_dfs, names)
```

```python
kde_plot('AdjO', train, test, filtered_dfs, names)
```

```python
kde_plot('AdjD', train, test, filtered_dfs, names)
```

The above charts highlight some of the differences between each data set:

**"Lenient" Data:** This data seems to be about halfway between the regular season data and the NCAA Tournament data. This makes sense because we're only cutting out the games between obviously not tournament quality teams, but this still leaves us with a lot of "barely" tournament quality games because the regular season data was distributed around a much lower (or higher depending on the feaute) center for each feature. Overall, I think this data could be a good choice for training models moving forward because it reduces the difference between the regular season without removing too many games.

**"Strict" Data:** This set appears to most closely resemble the feature distributions of the NCAA Tournament data. It looks to be a pretty good representation of tournament data, but the issue here could be the limited number of games in the training set. This method filtered out the regular season data to less than half the size of the "lenient" data. 

**"Seeded" Data:** I included this data set as well since it seemed like an easy rule to limit games to tournament quality. I mean why not just look at games between Tournament teams right? While it seems like an obvious solution, we can see from the feature distributions above that this ends up concentrating the features even more than the tournament data. This makes sense because small conferences often only have one team make the tournament by winning the conference tournament. These teams often have lower season stats/ratings and they make up most of the lower seeds in the tournament. The regular season just doesn't feature enough games between these types of teams and teams from bigger conferences and we end up with just games between NCAA Tournament teams from large conferences that represent most of the higher seeds and we get the even more concentrated distributions shown above.

```python
# Save each data set to csv
pd.concat([filtered_dfs[0], test]).to_csv('../../Data/Training/large_training.csv', index=False)
pd.concat([filtered_dfs[1], test]).to_csv('../../Data/Training/small_training.csv', index=False)
```

```python

```
