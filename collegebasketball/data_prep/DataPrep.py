import os
import pandas as pd


def update_kenpom(data):
    """
    Updates the school names in the Kenpom data to match the team names in the
    game scores data.

    Args:
        data(DataFrame): A pandas DataFrame with the Kenpom data.

    Returns:
        The same Kenpom data, but with updated school names.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    return update_names(data, 'Kenpom/TRank')


def update_TRank(data):
    """
    Updates the school names in the T-Rank data to match the team names in the
    game scores data.

    Args:
        data(DataFrame): A pandas DataFrame with the T-Rank data.

    Returns:
        The same T-Rank data, but with updated school names.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    return update_names(data, 'Kenpom/TRank')


def update_basic(data):
    """
    Updates the school names in the basic statistics data to match the team names
    in the game scores data.

    Args:
        data(DataFrame): A pandas DataFrame with the basic stats data.

    Returns:
        The same stats data, but with updated school names.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    # Update team names to remove strange characters
    for i, row in data.iterrows():
        if '\xa0' in row['Team']:
            team = row['Team'].replace('\xa0', '').replace('NCAA', '')
            data.iloc[int(i), data.columns.get_loc('Team')] = team

    return update_names(data, 'Stats')


# Actually updates the school names based on the schools.csv file.
def update_names(data, type):

    # Load school name data
    path = os.path.dirname(os.path.abspath(__file__)) + '/../../Data/Resources/schools.csv'
    school_names = pd.read_csv(path)
    stats = school_names.loc[:, type]
    scores = school_names.loc[:, 'Scores']

    # Create the dictionary from the two lists
    names_dict = dict(zip(stats, scores))

    # Go through schools in the data and replace school names in the dictionary with the right name
    for i, row in data.iterrows():
        if row['Team'] in names_dict:
            data.iloc[int(i), data.columns.get_loc('Team')] = names_dict[row['Team']]

    return data


# Checks that all of the school names in the stats datasets are correctly in the scores datasets
def check_for_missing_names(scores_df, stats_df, verbose=False):

    # Check that each school in the stats df is in the scores df
    missing_names = []
    for name in stats_df['Team'].unique():
        if name not in scores_df['Home'].values or name not in scores_df['Away'].values:
            missing_names.append(name)

    if verbose:
        print("Checked {0} schools and found {1} matches.".format(len(stats_df), len(missing_names)))
    return missing_names
