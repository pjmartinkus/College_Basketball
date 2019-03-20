import pandas as pd


def update_kenpom(data):

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    return update_names(data, 'Kenpom/TRank')


def update_TRank(data):

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    return update_names(data, 'Kenpom/TRank')


def update_basic(data):

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    # Update team names to remove strange characters
    for i, row in data.iterrows():
        if '\xa0' in row['Team']:
            team = row['Team'].replace('\xa0', '').replace('NCAA', '')
            data.iloc[int(i), data.columns.get_loc('Team')] = team

    return update_names(data, 'Stats')


def update_names(data, type):

    # Load school name data
    school_names = pd.read_csv('/Users/phil/Documents/Documents/College_Basketball/Data/School_Names/schools.csv')
    stats = school_names.loc[:, type]
    scores = school_names.loc[:, 'Scores']

    # Create the dictionary from the two lists
    names_dict = dict(zip(stats, scores))

    # Go through schools in the data and replace school names in the dictionary with the right name
    for i, row in data.iterrows():
        if row['Team'] in names_dict:
            data.iloc[int(i), data.columns.get_loc('Team')] = names_dict[row['Team']]

    return data
