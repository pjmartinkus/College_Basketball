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
regular_season = {}
march_madness = {}
this_year = 2019

# We need to first join datasets from the same year
for year in range(2002, this_year):
    
    # Load the scores datasets
    regular_season[year] = pd.read_csv(scores_path + str(year) + '_regular_season.csv')
    if year < this_year - 1:
        march_madness[year] = pd.read_csv(scores_path + str(year) + '_march.csv')
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
    assert len(cbb.check_for_missing_names(regular_season[year], kenpom_data[year], False)) == 0
    
    # TRank data starts in 2008 and the team name column is called school instead of team
    if year > 2007:
        data_TRank =  pd.read_csv('{0}TRank/{1}_TRank.csv'.format(path, year))
        TRank_data[year] = cbb.update_TRank(data_TRank)
        assert len(cbb.check_for_missing_names(regular_season[year], TRank_data[year], False)) == 0
        
    # Basic stats data starts in 2010
    if year > 2009:
        data_stats =  pd.read_csv('{0}SportsReference/{1}_stats.csv'.format(path, year))
        data_stats = data_stats.rename(index=str, columns={'School': 'Team'})
        stats_data[year] = cbb.update_basic(data_stats)
        assert len(cbb.check_for_missing_names(regular_season[year], stats_data[year], False)) == 0
```

```python
# Lets take a quick look at one of the datasets
kenpom_data[2013].head()
```

## Joining the Datasets

Now that the school names from each data set matches up, we can join the kenpom and score data to form a single csv file. Additionally, the march scores datasets contained some games that were from postseason tournament games other than the NCAA Tournamnet (for example NIT games are currently in the march scores data). Since, we joined the kenpom data (containing the team seeds) to the game scores, we can find and remove non-NCAA Tounament games from the march datasets by removing games that are not between teams with an NCAA Tournament seed. We will remove these games from the march datasets and add them in the regular season data.

```python
# Save the paths to the data 
save_path = '../Data/Combined_Data/'

# Save the joined tables in dictionaries
regular = {}
march = {}

# We need to first join datasets from the same year
for year in range(2002, this_year):
    
    # Join the dataframes to get kenpom for both home and away team
    regular[year] = pd.merge(regular_season[year], kenpom_data[year], left_on='Home', right_on='Team', sort=False)
    regular[year] = pd.merge(regular[year], kenpom_data[year], left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    
    # Do the same for the march data (No march data for this year yet)
    if year < this_year - 1:
        march[year] = pd.merge(march_madness[year], kenpom_data[year], left_on='Home', right_on='Team', sort=False)
        march[year] = pd.merge(march[year], kenpom_data[year], left_on='Away', right_on='Team', 
                                 suffixes=('_Home', '_Away'), sort=False)

        # Move non-tournament games to regular season data
        other_games = march[year][march[year]['Seed_Home'].isnull()]
        regular[year] = pd.concat([regular[year], other_games], ignore_index=True)
        march[year].drop(other_games.index, inplace=True)
    
    # Add a column to indicate the year
    regular[year].insert(0, 'Year', year)
    if year < this_year - 1:
        march[year].insert(0, 'Year', year)
        
# Combine the data for every year
regular_df = pd.concat(regular, ignore_index=True)
march_df = pd.concat(march, ignore_index=True)

# Save the data to csv files
regular_df.to_csv('{0}Kenpom.csv'.format(save_path), index=False)
march_df.to_csv('{0}Kenpom_march.csv'.format(save_path), index=False)
    
# Lets take a look at the data set
print("There are {} games in the Kenpom dataset.".format(len(regular_df)))
print("There are {} games in the march Kenpom dataset.".format(len(march_df)))
regular_df.head()
```

Now we will clean up the team names in the T-Rank data and join it with the game scores data. Additionally, we need to join these data sets with the team Kenpom statistics. This join is necessary because we need to use the Tournament seed attribute in order to clean up the march dataset to only include NCAA Tournament games. It will also be beneficial down the road, during feature generation, for us to have the Kenpom AdjEM and W/L stats for each team as a way to judge what outcome of a game is considered an upset.

```python
# Save the paths to the scores data 
save_path = '../Data/Combined_Data/'

# Save the joined tables in dictionaries
regular = {}
march = {}

# We need to first join datasets from the same year
for year in range(2008, this_year):
    
    # Get only the columns we need from the kenpom data
    kp = kenpom_data[year][['Team', 'AdjEM', 'Seed']]
    
    # Join the dataframes to get TRank data and kenpom (seed, adj_em) for both home and away team
    regular[year] = pd.merge(regular_season[year], TRank_data[year], left_on='Home', right_on='Team', sort=False)
    regular[year] = pd.merge(regular[year], TRank_data[year], left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    regular[year] = pd.merge(regular[year], kp, left_on='Home', right_on='Team', sort=False)
    regular[year] = pd.merge(regular[year], kp, left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    
    # Do the same for the march data (No march data for this year yet)
    if year < this_year - 1:
        march[year] = pd.merge(march_madness[year], TRank_data[year], left_on='Home', right_on='Team', sort=False)
        march[year] = pd.merge(march[year], TRank_data[year], left_on='Away', right_on='Team', 
                                 suffixes=('_Home', '_Away'), sort=False)
        march[year] = pd.merge(march[year], kp, left_on='Home', right_on='Team', sort=False)
        march[year] = pd.merge(march[year], kp, left_on='Away', right_on='Team', 
                                 suffixes=('_Home', '_Away'), sort=False)

        # Move non-tournament games to regular season data
        other_games = march[year][march[year]['Seed_Home'].isnull()]
        regular[year] = pd.concat([regular[year], other_games], ignore_index=True)
        march[year].drop(other_games.index, inplace=True)
    
    # Add a column to indicate the year
    regular[year].insert(0, 'Year', year)
    if year < this_year - 1:
        march[year].insert(0, 'Year', year)
    
# Combine the data for every year
regular_df = pd.concat(regular, ignore_index=True)
march_df = pd.concat(march, ignore_index=True)

# Save the data to csv files
regular_df.to_csv('{0}TRank.csv'.format(save_path), index=False)
march_df.to_csv('{0}TRank_march.csv'.format(save_path), index=False)
    
# Lets take a look at one of the data sets
print("There are {} games in the T-Rank dataset.".format(len(regular_df)))
print("There are {} games in the march T-Rank dataset.".format(len(march_df)))
regular_df.head()
```

Lastly, we will run the same process for the basic statistics as we did for the T-Rank data.

```python
# Save the paths to the scores data 
save_path = '../Data/Combined_Data/'

# Save the joined tables in dictionaries
regular = {}
march = {}

# We need to first join datasets from the same year
for year in range(2010, this_year):
    
    # Get only the columns we need from the kenpom data
    kp = kenpom_data[year][['Team', 'AdjEM', 'Seed', 'Wins', 'Losses']]
    
    # Join the dataframes to get basic statistics data and kenpom (seed, adj_em) for both home and away team
    regular[year] = pd.merge(regular_season[year], stats_data[year], left_on='Home', right_on='Team', sort=False)
    regular[year] = pd.merge(regular[year], stats_data[year], left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    regular[year] = pd.merge(regular[year], kp, left_on='Home', right_on='Team', sort=False)
    regular[year] = pd.merge(regular[year], kp, left_on='Away', right_on='Team', 
                             suffixes=('_Home', '_Away'), sort=False)
    
    # Do the same for the march data (No march data for this year yet)
    if year < this_year - 1:
        march[year] = pd.merge(march_madness[year], stats_data[year], left_on='Home', right_on='Team', sort=False)
        march[year] = pd.merge(march[year], stats_data[year], left_on='Away', right_on='Team', 
                                 suffixes=('_Home', '_Away'), sort=False)
        march[year] = pd.merge(march[year], kp, left_on='Home', right_on='Team', sort=False)
        march[year] = pd.merge(march[year], kp, left_on='Away', right_on='Team', 
                                 suffixes=('_Home', '_Away'), sort=False)

        # Move non-tournament games to regular season data
        other_games = march[year][march[year]['Seed_Home'].isnull()]
        regular[year] = pd.concat([regular[year], other_games], ignore_index=True)
        march[year].drop(other_games.index, inplace=True)
    
    # Add a column to indicate the year
    regular[year].insert(0, 'Year', year)
    if year < this_year - 1:
        march[year].insert(0, 'Year', year)
    
# Combine the data for every year
regular_df = pd.concat(regular, ignore_index=True)
march_df = pd.concat(march, ignore_index=True)

# Save the data to csv files
regular_df.to_csv('{0}Basic.csv'.format(save_path), index=False)
march_df.to_csv('{0}Basic_march.csv'.format(save_path), index=False)
    
# Lets take a look at one of the data sets
print("There are {} games in the regular season basic statistics dataset.".format(len(regular_df)))
print("There are {} games in the march basic statistics dataset.".format(len(march_df)))
regular_df.head()
```
