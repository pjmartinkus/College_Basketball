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

## Blocking

```python
# Import packages
import sys
sys.path.append('../')

import datetime
import pandas as pd

import collegebasketball as cbb
cbb.__version__
```

```python
# Initialize some variables
dataset_names = ['kenpom', 'TRank', 'stats', 'all']
dataset_types = ['season', 'season_blocked', 'march', 'seed']
drop_cols = ['Seed', 'Seed_Fav', 'Seed_Diff']
path = '../Data/Training/'
rows = dict()
data = dict()

# Process each data source
for name in dataset_names:
    
    # Load in datasets
    for dt in dataset_types[0:-1]:
        data[name + dt] = pd.read_csv('{0}{1}_{2}.csv'.format(path, name, dt))
    
    # Apply seed based blocking rule
    df = data[name + 'season'].copy()
    data[name + 'seed'] = df[(df['Seed'].notnull()) & (df['Seed_Fav'].notnull())]
    
    # Drop extra columns
    for dt in dataset_types:
        data[name + dt] = data[name + dt].drop(['Seed', 'Seed_Fav', 'Seed_Diff'], axis=1)
    
    # Save sizes
    rows[name] = [name]
    for dt in dataset_types:
        rows[name].append(len(data[name + dt]))

size_df = pd.DataFrame.from_dict(rows, orient='index').drop(0, axis=1)
size_df.columns = pd.Index(dataset_types)
print('Dataset Sizes:')
size_df
```

```python
# Process each datasets
dataset_types.remove('march')
for name in dataset_names:
    print(name + ':')
    for dt in dataset_types:
        
        # See how similar each dataset is to the march data
        mcc, f1 = cbb.covariate_shift(data[name + dt], data[name + 'march'])
        print('   {0}: mcc = {1}, f1 = {2}'.format(dt, mcc, f1))
```

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
```

```python
knn =  KNeighborsClassifier()
dt = DecisionTreeClassifier(min_samples_leaf=5)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=5)
log = LogisticRegression(penalty='l1', C=10)

cls = [knn, dt, rf, log]
cl_names = ['KNN', 'Decision Tree', 'Random Forest', 'Logistic Regression']
exclude = ['Favored', 'Underdog', 'Year', 'Label']
```

```python
train = data['kenpomseason'].copy()
test = data['kenpommarch'].copy()
cols = train.columns.to_series().reset_index(drop=True)
stats = cols[cols.apply(lambda x: 'Rank' not in x and
                                  'Fav' not in x and
                                  'Diff' not in x)].tolist()
rank_cols = cols[cols.apply(lambda x: 'Rank' in x or 'Label' in x or 'Win' in x)].tolist()
value_cols = cols[cols.apply(lambda x: 'Rank' not in x)].tolist()
diff_cols = cols[cols.apply(lambda x: 'Diff' not in x)].tolist()
```

```python
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
cbb.evaluate(train[rank_cols], test, ['Label'], cls, cl_names)
```

```python
cbb.evaluate(train[value_cols], test, exclude, cls, cl_names)
```

```python
cbb.evaluate(train[diff_cols], test, exclude, cls, cl_names)
```

```python
# Extract the actual statistics in this dataset
train = data['kenpomseason'].copy()
test = data['kenpommarch'].copy()
cols = train.columns.drop(exclude).to_series().reset_index(drop=True)
stats = cols[cols.apply(lambda x: 'Rank' not in x and
                                  'Fav' not in x and
                                  'Diff' not in x)]
types = ['', '_Fav', '_Diff']

pca_objects = dict()
train_pca = list()
test_pca = list()

# Apply pca to columns for each stat
for stat in stats:
#     print(stat)
    
    for t in types:
        
        if stat not in ['Win_Loss', 'AdjEM']:
            stat_columns = [stat + t, stat + ' Rank' + t]
        elif stat == 'Win_Loss':
            stat_columns = [stat + t]
        else:
            stat_columns = ['AdjEM' + t, 'Rank' + t]
        
        pca = PCA(n_components=1)
        pca_objects[stat] = pca
        
#         print(stat + t + '\t' + str(list(stat_columns)))

        train_pca.append(pd.DataFrame(pca.fit_transform(data['kenpomseason'][stat_columns]), columns=[stat + t]))
        test_pca.append(pd.DataFrame(pca.transform(data['kenpommarch'][stat_columns]), columns=[stat + t]))
```

```python
train = pd.concat(train_pca, axis=1)
test = pd.concat(test_pca, axis=1)
train['Label'] = data['kenpomseason']['Label']
test['Label'] = data['kenpommarch']['Label']

# cbb.evaluate(train, test, ['Label'], cls, cl_names)
```

```python
train = data['kenpomseason']
test = data['kenpommarch']
print('Kenpom Data:')
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
print('Kenpom Seed:')
train = data['kenpomseed']
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
train = data['TRankseason']
test = data['TRankmarch']
print('T-Rank Data:')
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
print('T-Rank Seed:')
train = data['TRankseed']
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
train = data['statsseason']
test = data['statsmarch']
print('Stats Data:')
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
print('Stats Seed:')
train = data['statsseed']
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
train = data['allseason'].drop(['Seed_Diff_x', 'Seed_Diff_y'], axis=1)
test = data['allmarch'].drop(['Seed_Diff_x', 'Seed_Diff_y'], axis=1)
print('All:')
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
train = data['allseed'].drop(['Seed_Diff_x', 'Seed_Diff_y'], axis=1)
print('All:')
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
train = data['allseason'].copy().drop(['Seed_Diff_x', 'Seed_Diff_y'], axis=1)
test = data['allmarch'].copy().drop(['Seed_Diff_x', 'Seed_Diff_y'], axis=1)
cols = train.columns.to_series().reset_index(drop=True)
stats = cols[cols.apply(lambda x: 'Rank' not in x and
                                  'Fav' not in x and
                                  'Diff' not in x)].tolist()
rank_cols = cols[cols.apply(lambda x: 'Rank' in x or 'Label' in x or 'Win' in x)].tolist()
value_cols = cols[cols.apply(lambda x: 'Rank' not in x)].tolist()
diff_cols = cols[cols.apply(lambda x: 'Diff' not in x)].tolist()
```

```python
cbb.evaluate(train, test, exclude, cls, cl_names)
```

```python
cbb.evaluate(train[rank_cols], test, ['Label'], cls, cl_names)
```

```python
cbb.evaluate(train[value_cols], test, exclude, cls, cl_names)
```

```python
cbb.evaluate(train[diff_cols], test, exclude, cls, cl_names)
```

```python
only_value_cols = cols[cols.apply(lambda x: 'Diff' not in x and
                                            'Rank' not in x)].tolist()
cbb.evaluate(train[only_value_cols], test, exclude, cls, cl_names)
```

```python
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
```

```python
[1, 2, 3].remove(3)
```

```python
def rank_features(train, test, exclude, model):
    
    # Intialize some variables
    curr_features = list()
    feature_metrics = list()
    curr_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    feature_names = [x for x in train.columns if x not in exclude]
    
    for _ in range(len(feature_names)):
        #print('Outer loop:')
        
        max_diff = 0
        max_metric = None
        
        # Test a model with the current features plus one new feature
        for feature in feature_names:
            
            #print('\t' + feature + ':')
            
            model.fit(train[curr_features + [feature]], train[['Label']].values.ravel())
            predictions = model.predict(test[curr_features + [feature]])
            
            metrics = {'precision': precision_score(test['Label'], predictions),
                       'recall': recall_score(test['Label'], predictions), 
                       'f1': f1_score(test['Label'], predictions)}
            metrics['diff'] = np.prod([metrics[key] - curr_metrics[key] for key in metrics.keys()])
            
            #print('\t' + str(metrics))

            if metrics['diff'] > max_diff:
                max_diff = metrics['diff']
                max_metric = feature
                max_values = metrics
                
                #print('\t\tnew max')
            
        # If no additional feature improve the score, stop
        if max_metric is None:
             break
        else:
            curr_features.append(max_metric)
            curr_metrics = metrics
            feature_names.remove(feature)
            feature_metrics.append(max_values)
    
    return curr_features, feature_metrics
```

```python
curr_features, feature_metrics = rank_features(train, test, exclude, cls[-1])
curr_features, feature_metrics
```

```python

```
