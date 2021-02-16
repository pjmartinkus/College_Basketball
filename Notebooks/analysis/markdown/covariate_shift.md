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
 
**TODO:** explain general idea of covariate shift analysis and how we can use it to better understand if our regular season training data is actually representative of tournament games.

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




## TODO: 

In general we can see that the season and tournament data look very different.

Next we'll try an zero in on some features that show the biggest difference.

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

## TODO:

Next plot distributions of most important features

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

## TODO:

Try to come up with some blocking rules that don't remove too much tournament worthy data from the regular season data

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

tuning_df = pd.read_csv('../../Data/analysis/cov_shift.csv')
        
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

## TODO:

Try using some blocking rules and look at covariate shift and plots

For reference - original scores:
* Kenpom: auc = 0.9153326865550974, f1 = 0.9140271493212668
* T-Rank: auc = 0.910958904109589, f1 = 0.9090909090909091
* Basic Stats: auc = 0.92126343897126, f1 = 0.9277566539923955
* All Data: auc = 0.9076923076923077, f1 = 0.9153846153846154

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

```python

```
