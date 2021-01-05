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

# Retrieving College Basketball Data

In this notebook, I retrieve the scores and kenpom data for all teams from 2002 to 2017. The data is saved in csv files so that it can be accessed later by other notebooks.

```python
# Import packages
import sys
sys.path.append('../')

import datetime
import pandas as pd
import collegebasketball as cbb
cbb.__version__
```

## Getting the Scores Data

The scores are from https://www.sports-reference.com/cbb/. Below shows the code I used to download all the scores in college basketball from 2002 to 2017.

For each season, I create two different csv files. One contains the regular season data and the other file contains just the scores from the NCAA tournament. I will later use the Tournment scores to evaluate the performance of the models and seperating the data at this stage will simplify things later on.

```python
# This list contains the starting dates of march madness for each year
march_start_dates = [12, 18, 16, 15, 14, 13, 18, 17, 16, 15, 13, 19, 18, 17, 15, 14, 19]

# The location where the files will be saved
path = '../Data/Scores/'
```

```python
# We will be creating a csv file for each regular season and tournament from 2002 to 2019
for year in range(2002, 2003):

    # Set up the starting and ending dates of the regular season and march madness
    start_regular = datetime.date(year - 1, 11, 1)
    end_regular = datetime.date(year, 3, march_start_dates[year - 2003] - 1)
    
    start_march = datetime.date(year, 3, march_start_dates[year - 2003])
    end_march = datetime.date(year, 4, 10)
    
    # Set up the path for this years scores
    path_regular = path + str(year) + '_regular_season.csv'
    path_march = path + str(year) + '_march.csv'

    # Create and save the csv files for the regular season and march madness data for the year
    cbb.load_scores_dataframe(start_date=start_regular, end_date=end_regular, csv_file_path=path_regular)
    cbb.load_scores_dataframe(start_date=start_march, end_date=end_march, csv_file_path=path_march)
```

```python
# Load a dataset to take an initial look
file_path = path + '2003_march.csv'
data = pd.read_csv(file_path)

data.head()
```

```python
# Let's take a look at all the games involving Marquette during the 2003 Tournament
pd.concat([data[data['Home'] == 'Marquette'], data[data['Away'] == 'Marquette']])
```

## Getting Basic Team Stats

The teams stats data is also from https://www.sports-reference.com/cbb/. This data contains basic basketball statistics for each team at the end of each season. These stats will later be used to train the model and evaluate teams.

```python
# The location where the files will be saved
path = '../Data/SportsReference/'

# We will be creating a csv file of data for each season from 2003 to 2019
for year in range(2010, 2020):
    
    # Set the path for the current year data
    stats_path = path + str(year) + '_stats.csv'
    
    # Save the basic stats data into a csv file
    cbb.load_stats_dataframe(year=year, csv_file_path=stats_path)
```

```python
# Load some data to take a look
stats_path = path + '2013_stats.csv'
data = pd.read_csv(stats_path)

data.head()
```

## Getting the Kenpom Data

The kenpom data is from https://kenpom.com. This website displays advanced stats for each team in the NCAA. These stats will later be used to train the model and evaluate teams.

```python
# The location where the files will be saved
path = '../Data/Kenpom/'

# We will be creating a csv file of kenpom data for each season from 2002 to 2019
for year in range(2002, 2020):
    
    # Set the path for the current year data
    kp_path = path + str(year) + '_kenpom.csv'
    
    # Save the kenpom data into a csv file
    cbb.load_kenpom_dataframe(year=year, csv_file_path=kp_path)
```

```python
# Load some data to take a look
kp_path = path + '2003_kenpom.csv'
data = pd.read_csv(kp_path)

data.head()
```

```python
# Let's take a look at Marquette's kenpom numbers for 2003
data[data['Team'] == 'Marquette']
```

## Getting the T-Rank Data

The T-Rank data is from http://www.barttorvik.com/#. This website displays advanced stats for each team in the NCAA. These stats will later be used to train the model and evaluate teams.

```python
# The location where the files will be saved
path = '../Data/TRank/'

# We will be creating a csv file of data for each season from 2008 to 2019
for year in range(2008, 2020):
    
    # Set the path for the current year data
    TRank_path = path + str(year) + '_TRank.csv'
    
    # Save the T-Rank data into a csv file
    cbb.load_TRank_dataframe(year=year, csv_file_path=TRank_path)
```

```python
# Load some data to take a look
TRank_path = path + '2008_TRank.csv'
data = pd.read_csv(TRank_path)

data.head()
```

```python
# Let's take a look at Marquette's kenpom numbers for 2008
data[data['Team'] == 'Marquette']
```