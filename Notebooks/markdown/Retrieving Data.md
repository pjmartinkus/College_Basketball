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

For each season, I create a csv file with all of the game scores for that season. Each record contains the team names, score and the tournament the game was played in (if applicable).

```python
# The location where the files will be saved
path = '../Data/Scores/'
year = 2024
```

```python
# Load regular season scores data from the current season. This data set goes back to 2002 (though you might want to ignore 2020)
# and we have previously retrieved that data in this project
start = datetime.date(year - 1, 11, 1)
end = datetime.date(year, 3, 18)

# Set up the path for this years scores
path_regular = path + str(year) + '_season.csv'

# Create and save the csv files for the regular season and march madness data for the year
cbb.load_scores_dataframe(start, end, csv_file_path=path_regular)
```

```python
# Load a dataset to take an initial look
file_path = path + '2024_season.csv'
data = pd.read_csv(file_path)

data.head()
```

```python
# Let's take a look at all the games involving Marquette during the 2003 Tournament
data = cbb.filter_tournament(pd.read_csv(path + '2003_season.csv'))
data[(data['Home'] == 'Marquette') | (data['Away'] == 'Marquette')]
```

## Getting Basic Team Stats

The teams stats data is also from https://www.sports-reference.com/cbb/. This data contains basic basketball statistics for each team at the end of each season. These stats will later be used to train the model and evaluate teams.

```python
# The location where the files will be saved
path = '../Data/SportsReference/'

# Like before we will add this season to the data going back to 2002
stats_path = path + str(year) + '_stats.csv'
    
# Save the basic stats data into a csv file
cbb.load_stats_dataframe(year=year, csv_file_path=stats_path)
```

```python
# Load some data to take a look
stats_path = path + '2024_stats.csv'
data = pd.read_csv(stats_path)

data.head()
```

## Getting the Kenpom Data

The kenpom data is from https://kenpom.com. This website displays advanced stats for each team in the NCAA. These stats will later be used to train the model and evaluate teams.

```python
# The location where the files will be saved
path = '../Data/Kenpom/'

# Since the kenpom data has become harder to scrap we'll manually save the html to a file
kp_input_path = '../Data/kenpom_html/kenpom_2024.html'

# This data also goes back to 2002, but we'll just retrieve the current season now
kp_path = path + str(year) + '_kenpom.csv'
    
# Save the kenpom data into a csv file
cbb.load_kenpom_dataframe(year=year, csv_file_path=kp_path, input_file_path=kp_input_path)
```

```python
# Load some data to take a look
kp_path = path + '2024_kenpom.csv'
data = pd.read_csv(kp_path)

data.head()
```

```python
# Let's take a look at Marquette's kenpom numbers for 2024
data[data['Team'] == 'Marquette']
```

## Getting the T-Rank Data

The T-Rank data is from http://www.barttorvik.com/#. This website displays advanced stats for each team in the NCAA. These stats will later be used to train the model and evaluate teams.

```python
# The location where the files will be saved
path = '../Data/TRank/'

# This data goes back to 2008 but we'll just get this season
TRank_path = path + str(year) + '_TRank.csv'
    
# Save the T-Rank data into a csv file
cbb.load_TRank_dataframe(year=year, csv_file_path=TRank_path)
```

```python
# Load some data to take a look
TRank_path = path + '2024_TRank.csv'
data = pd.read_csv(TRank_path)

data.head()
```

```python
# Let's take a look at Marquette's numbers for 2024
data[data['Team'] == 'Marquette']
```

```python

```
