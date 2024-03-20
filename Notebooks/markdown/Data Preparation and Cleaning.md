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
year = 2024

# Load the scores datasets
scores_data = pd.read_csv(scores_path + str(year) + '_season.csv')
```

## Cleaning the Data

Next, we need to edit the school names in the kenpom, basic stats and T-Rank datasets to ensure that they match up with the school names from the scores dataset. We will verify that the names match using the `cbb.check_for_missing_names` command. It checks that each school name in the given team statistics dataset (kenpom, basic stats or T-Rank) is present in the game scores dataset.

```python
# The location where the files will be saved
path = '../Data/'
    
# Load this year's data and clean up the school names to match up with scores data
kenpom_data = pd.read_csv('{0}Kenpom/{1}_kenpom.csv'.format(path, year))
kenpom_data = cbb.update_kenpom(kenpom_data)
assert len(cbb.check_for_missing_names(scores_data, kenpom_data, False)) == 0

# TRank data
TRank_data =  pd.read_csv('{0}TRank/{1}_TRank.csv'.format(path, year))
TRank_data = cbb.update_TRank(TRank_data)
assert len(cbb.check_for_missing_names(scores_data, TRank_data, False)) == 0

# Basic stats data
stats_data =  pd.read_csv('{0}SportsReference/{1}_stats.csv'.format(path, year))
stats_data = stats_data.rename(index=str, columns={'School': 'Team'})
stats_data = cbb.update_basic(stats_data)
assert len(cbb.check_for_missing_names(scores_data, stats_data, False)) == 0
```

```python
# Lets take a quick look at one of the datasets
kenpom_data.head()
```

## Joining the Datasets

Now that the school names from each data set matches up, we can join the kenpom and score data to form a single csv file. 

```python
# Save the paths to the data 
save_path = '../Data/Combined_Data/Kenpom.csv'
    
# Join the dataframes to get kenpom for both home and away team
kenpom_df = pd.merge(scores_data, kenpom_data, left_on='Home', right_on='Team', sort=False)
kenpom_df = pd.merge(kenpom_df, kenpom_data, left_on='Away', right_on='Team', 
                     suffixes=('_Home', '_Away'), sort=False)

# Add a column to indicate the year
kenpom_df.insert(0, 'Year', year)
        
# Combine the data for every year and save to csv
all_kenpom = pd.read_csv(save_path)
kenpom_df = pd.concat([all_kenpom, kenpom_df])
kenpom_df.to_csv(save_path, index=False)
    
# Lets take a look at the data set
print("There are {} games in the Kenpom dataset.".format(len(kenpom_df)))
print("There are {} NCAA Tournament games in the Kenpom dataset.".format(len(cbb.filter_tournament(kenpom_df))))
kenpom_df.head()
```

Now we will clean up the team names in the T-Rank data and join it with the game scores data. Additionally, we need to join these data sets with the team Kenpom statistics. This join is necessary because we need to use the Tournament seed attribute in order to clean up the march dataset to only include NCAA Tournament games. It will also be beneficial down the road, during feature generation, for us to have the Kenpom AdjEM and W/L stats for each team as a way to judge what outcome of a game is considered an upset.

```python
save_path = '../Data/Combined_Data/TRank.csv'

# Get only the columns we need from the kenpom data
kp = kenpom_data[['Team', 'AdjEM', 'Seed']]

# Join the dataframes to get TRank data and kenpom (seed, adj_em) for both home and away team
TRank_df = pd.merge(scores_data, TRank_data, left_on='Home', right_on='Team', sort=False)
TRank_df = pd.merge(TRank_df, TRank_data, left_on='Away', right_on='Team', 
                         suffixes=('_Home', '_Away'), sort=False)
TRank_df = pd.merge(TRank_df, kp, left_on='Home', right_on='Team', sort=False)
TRank_df = pd.merge(TRank_df, kp, left_on='Away', right_on='Team', 
                    suffixes=('_Home', '_Away'), sort=False)

# Add a column to indicate the year
TRank_df.insert(0, 'Year', year)

# T-Rank has introduced a new column - for now we'll just drop it but should include in future
drop_cols = ['3PR_Home', '3PR Rank_Home', '3PRD_Home', '3PRD Rank_Home', '3PR_Away', '3PR Rank_Away', 
             '3PRD_Away', '3PRD Rank_Away']
TRank_df = TRank_df.drop(drop_cols, axis=1)
    
# Combine the data for every year and save to csv
all_TRank = pd.read_csv(save_path)
all_TRank.rename(columns={'Team_Home.1': 'Team_Home', 'Team_Away.1': 'Team_Away'}, inplace=True)
TRank_df = pd.concat([all_TRank, TRank_df])
TRank_df.to_csv(save_path, index=False)
    
# Lets take a look at one of the data sets
print("There are {} games in the T-Rank dataset.".format(len(TRank_df)))
print("There are {} NCAA Tournament games in the T-Rank dataset.".format(len(cbb.filter_tournament(TRank_df))))
TRank_df.head()
```

Lastly, we will run the same process for the basic statistics as we did for the T-Rank data.

```python
save_path = '../Data/Combined_Data/Basic.csv'
    
# Get only the columns we need from the kenpom data
kp = kenpom_data[['Team', 'AdjEM', 'Seed', 'Wins', 'Losses']]

# Join the dataframes to get basic statistics data and kenpom (seed, adj_em) for both home and away team
basic_df = pd.merge(scores_data, stats_data, left_on='Home', right_on='Team', sort=False)
basic_df = pd.merge(basic_df, stats_data, left_on='Away', right_on='Team', 
                    suffixes=('_Home', '_Away'), sort=False)
basic_df = pd.merge(basic_df, kp, left_on='Home', right_on='Team', sort=False)
basic_df = pd.merge(basic_df, kp, left_on='Away', right_on='Team', 
                    suffixes=('_Home', '_Away'), sort=False)

# Add a column to indicate the year
basic_df.insert(0, 'Year', year)
    
# Combine the data for every year and save to csv
all_basic = pd.read_csv(save_path)
all_basic.rename(columns={'Team_Home.1': 'Team_Home', 'Team_Away.1': 'Team_Away'}, inplace=True)
basic_df = pd.concat([all_basic, basic_df])
basic_df.to_csv(save_path, index=False)
    
# Lets take a look at one of the data sets
print("There are {} games in the regular season basic statistics dataset.".format(len(basic_df)))
print("There are {} NCAA tournament games in the basic statistics dataset.".format(len(cbb.filter_tournament(basic_df))))
basic_df.head()
```

```python

```
