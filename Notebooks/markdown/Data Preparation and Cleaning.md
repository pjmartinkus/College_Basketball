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

# Data Preparation and Cleaning

In this notebook, I clean the datasets and combine them into a single csv file that can be used later for feature generation.

```python
# Import packages
import sys
sys.path.append('../../College_Basketball')

import pandas as pd
import collegebasketball as cbb
cbb.__version__
```

## Load in the Game Scores Data

First, we will load in the games scores data from csv files we created earlier. Later, we'll join this data to the team stats datasets. 

```python
# Location of the data
scores_path = '../Data/Scores/'

# Initialize some variables
scores_data = {}
this_year = 2019

# Load the scores datasets
for year in range(2002, this_year):
    scores_data[year] = pd.read_csv(scores_path + str(year) + '_season.csv')
```

## Cleaning the Data

Next, we need to edit the school names in the kenpom, basic stats and T-Rank datasets to ensure that they match up with the school names from the scores dataset. We will verify that the names match using the `cbb.check_for_missing_names` command. It checks that each school name in the given team statistics dataset (kenpom, basic stats or T-Rank) is present in the game scores dataset.

```python
# The location where the files will be saved
path = '../Data/'

# Store a dataframe of kenpom data for each year in a list
kenpom_data = {}
TRank_data = {}
stats_data = {}

# We need to clean each statistics data set
for year in range(2002, this_year):
    
    # Load this year's data and clean up the school names to match up with scores data
    data_kenpom = pd.read_csv('{0}Kenpom/{1}_kenpom.csv'.format(path, year))
    kenpom_data[year] = cbb.update_kenpom(data_kenpom)
    assert len(cbb.check_for_missing_names(scores_data[year], kenpom_data[year], False)) == 0
    
    # TRank data starts in 2008
    if year > 2007:
        data_TRank =  pd.read_csv('{0}TRank/{1}_TRank.csv'.format(path, year))
        TRank_data[year] = cbb.update_TRank(data_TRank)
        assert len(cbb.check_for_missing_names(scores_data[year], TRank_data[year], False)) == 0
        
    # Basic stats data starts in 2010 and the team name column is called school instead of team
    if year > 2009:
        data_stats =  pd.read_csv('{0}SportsReference/{1}_stats.csv'.format(path, year))
        data_stats = data_stats.rename(index=str, columns={'School': 'Team'})
        stats_data[year] = cbb.update_basic(data_stats)
        assert len(cbb.check_for_missing_names(scores_data[year], stats_data[year], False)) == 0
```

```python
# Lets take a quick look at one of the datasets
kenpom_data[2013].head()
```

## Joining the Datasets

Now that the school names from each data set matches up, we can join the kenpom and score data to form a single csv file. 

```python
# Save the paths to the data 
save_path = '../Data/Combined_Data/'

# Save the joined tables in dictionaries
data = {}

# We need to first join datasets from the same year
for year in range(2002, this_year):
    
    # Join the dataframes to get kenpom for both home and away team
    data[year] = pd.merge(scores_data[year], kenpom_data[year], left_on='Home', right_on='Team', sort=False)
    data[year] = pd.merge(data[year], kenpom_data[year], left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    
    # Add a column to indicate the year
    data[year].insert(0, 'Year', year)
        
# Combine the data for every year and save to csv
kenpom_df = pd.concat(data, ignore_index=True)
kenpom_df.to_csv('{0}Kenpom.csv'.format(save_path), index=False)
    
# Lets take a look at the data set
print("There are {} games in the Kenpom dataset.".format(len(kenpom_df)))
print("There are {} NCAA Tournament games in the Kenpom dataset.".format(len(cbb.filter_tournament(kenpom_df))))
kenpom_df.head()
```

Now we will clean up the team names in the T-Rank data and join it with the game scores data. Additionally, we need to join these data sets with the team Kenpom statistics. This join is necessary because we need to use the Tournament seed attribute in order to clean up the march dataset to only include NCAA Tournament games. It will also be beneficial down the road, during feature generation, for us to have the Kenpom AdjEM and W/L stats for each team as a way to judge what outcome of a game is considered an upset.

```python
save_path = '../Data/Combined_Data/'
data = {}

# We need to first join datasets from the same year
for year in range(2008, this_year):
    
    # Get only the columns we need from the kenpom data
    kp = kenpom_data[year][['Team', 'AdjEM', 'Seed']]
    
    # Join the dataframes to get TRank data and kenpom (seed, adj_em) for both home and away team
    data[year] = pd.merge(scores_data[year], TRank_data[year], left_on='Home', right_on='Team', sort=False)
    data[year] = pd.merge(data[year], TRank_data[year], left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    data[year] = pd.merge(data[year], kp, left_on='Home', right_on='Team', sort=False)
    data[year] = pd.merge(data[year], kp, left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    
    # Add a column to indicate the year
    data[year].insert(0, 'Year', year)
    
# Combine the data for every year and save to csv
TRank_df = pd.concat(data, ignore_index=True)
TRank_df.to_csv('{0}TRank.csv'.format(save_path), index=False)
    
# Lets take a look at one of the data sets
print("There are {} games in the T-Rank dataset.".format(len(TRank_df)))
print("There are {} NCAA Tournament games in the T-Rank dataset.".format(len(cbb.filter_tournament(TRank_df))))
TRank_df.head()
```

Lastly, we will run the same process for the basic statistics as we did for the T-Rank data.

```python
save_path = '../Data/Combined_Data/'
data = {}

# We need to first join datasets from the same year
for year in range(2010, this_year):
    
    # Get only the columns we need from the kenpom data
    kp = kenpom_data[year][['Team', 'AdjEM', 'Seed', 'Wins', 'Losses']]
    
    # Join the dataframes to get basic statistics data and kenpom (seed, adj_em) for both home and away team
    data[year] = pd.merge(scores_data[year], stats_data[year], left_on='Home', right_on='Team', sort=False)
    data[year] = pd.merge(data[year], stats_data[year], left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    data[year] = pd.merge(data[year], kp, left_on='Home', right_on='Team', sort=False)
    data[year] = pd.merge(data[year], kp, left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    
    # Add a column to indicate the year
    data[year].insert(0, 'Year', year)
    
# Combine the data for every year and save to csv
basic_df = pd.concat(data, ignore_index=True)
basic_df.to_csv('{0}Basic.csv'.format(save_path), index=False)
    
# Lets take a look at one of the data sets
print("There are {} games in the regular season basic statistics dataset.".format(len(basic_df)))
print("There are {} NCAA tournament games in the basic statistics dataset.".format(len(cbb.filter_tournament(basic_df))))
basic_df.head()
```
