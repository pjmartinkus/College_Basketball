from collegebasketball.features.Features import feature_difference, win_loss, teams
import pandas as pd


def gen_features(data, feature_names):
    """
    Generates the features given by feature names from the given data set. For each
    of the specified feature names, a feature is created to reflect the favored team's
    stat for the feature, the underdog's stat and the difference between the two.

    Args:
        data(DataFrame): Input data to block.
        feature_names(list): List of feature names to create. These names should be the
                             names of attributes in the input data table.

    Returns:
        A new pandas DataFrame that includes the team names, year, and features for each
        of the specified feature names.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas DataFrame.')

    feature_vecs = []

    # Create the column names list
    cols = ['Favored', 'Underdog', 'Year', 'Tournament', 'Win_Loss_Fav', 'Win_Loss', 'Win_Loss_Diff']
    for feat in feature_names:
        cols.append(feat + '_Fav')
        cols.append(feat)
        cols.append(feat + '_Diff')
    columns = data.columns
    if 'Home_Score' in columns:
        cols.append('Label')

    # Generate the features for each row
    for row in data.itertuples(index=False):

        # Generate the initial data
        vec = dict()
        vec['Favored'], vec['Underdog'] = teams(row, columns)
        vec['Year'] = row[columns.get_loc('Year')]
        vec['Tournament'] = row[columns.get_loc('Tournament')]

        # Add the win loss features
        vec['Win_Loss_Fav'], vec['Win_Loss'], vec['Win_Loss_Diff'] = win_loss(row, columns)

        # Generate difference features
        for feat in feature_names:
            vec[feat + '_Fav'], vec[feat], vec[feat+'_Diff'] = feature_difference(row, columns, feat)

        # Add the label
        if 'Home_Score' in columns:
            if row[columns.get_loc('AdjEM_Home')] > row[columns.get_loc('AdjEM_Away')]:
                # Home team was favored and won
                if row[columns.get_loc('Home_Score')] > row[columns.get_loc('Away_Score')]:
                    vec['Label'] = 0
                # Away team upsets the home team
                else:
                    vec['Label'] = 1
            else:
                # Home team upset the away team
                if row[columns.get_loc('Home_Score')] > row[columns.get_loc('Away_Score')]:
                    vec['Label'] = 1
                # Away team was favored and won
                else:
                    vec['Label'] = 0

        # Append the new vector to the list
        feature_vecs.append(vec)

    # Return the dataframe
    return pd.DataFrame(feature_vecs, columns=cols)


def gen_kenpom_features(data):
    """
    Generates the features for the Kenpom data set.

    Args:
        data(DataFrame): Input data to block.

    Returns:
        A new pandas DataFrame that includes the team names, year, and features for
        the Kenpom data set.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas DataFrame.')

    # Names of all the features for Kenpom
    feature_names = ['Rank', 'Seed', 'AdjEM', 'AdjO', 'AdjO Rank', 'AdjD', 'AdjD Rank', 'AdjT',
                     'AdjT Rank', 'Luck', 'Luck Rank', 'OppAdjEM', 'OppAdjEM Rank', 'OppO',
                     'OppO Rank', 'OppD', 'OppD Rank', 'NCSOS AdjEM', 'NCSOS AdjEM Rank']

    return gen_features(data, feature_names)


def gen_TRank_features(data):
    """
    Generates the features for the T-Rank data set. The Kenpom data set is also
    necessary for this function because it is used to determing which team is
    considered the underdog and which is considered the favorite.

    Args:
        data(DataFrame): Input data to block.

    Returns:
        A new pandas DataFrame that includes the team names, year, and features for
        the T-Rank data set.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
        AssertionError: If kenpom_data is not of type pandas DataFrame.
    """

    # Check that data and kenpom_data are dataframes
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas DataFrame.')

    # Names of the features for T-Rank
    feature_names = ['Rk', 'Seed', 'AdjOE', 'AdjOE Rank', 'AdjDE', 'AdjDE Rank', 'Barthag',
                     'EFG%', 'EFG% Rank', 'EFGD%', 'EFGD% Rank', 'TOR', 'TOR Rank',
                     'TORD', 'TORD Rank', 'ORB', 'ORB Rank', 'DRB', 'DRB Rank',
                     'FTR', 'FTR Rank', 'FTRD', 'FTRD Rank', '2P%', '2P% Rank',
                     '2P%D', '2P%D Rank', '3P%D', '3P%D Rank', 'Adj T.', 'Adj T. Rank',
                     'WAB', 'WAB Rank', 'AdjEM']

    # Now that the data is in the correct format, create features
    return gen_features(data, feature_names)


def gen_basic_features(data):
    """
    Generates the features for the basic team stats data set. The Kenpom data
    set is necessary for this function because it is used to determing which
    team is considered the underdog and which is considered the favorite.

    Args:
        data(DataFrame): Input data to block.
        kenpom_data(DataFrame): The Kenpom data for the teams for the same year as
                                the input data set.

    Returns:
        A new pandas DataFrame that includes the team names, year, and features for
        the basic stats data set.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
        AssertionError: If kenpom_data is not of type pandas DataFrame.
    """

    # Check that data and kenpom_data are dataframes
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas DataFrame.')

    # Names of the features for the basic stats
    feature_names = ['Tm.', 'Seed', 'Opp.', 'MP', 'FG', 'FG_opp', 'FGA', 'FGA_opp', 'FG%',
                     'FG%_opp', '3P', '3P_opp', '3PA', '3PA_opp', '3P%', '3P%_opp',
                     'FT', 'FT_opp', 'FTA', 'FTA_opp', 'FT%', 'FT%_opp', 'ORB',
                     'ORB_opp', 'TRB', 'TRB_opp', 'AST', 'AST_opp', 'STL', 'STL_opp',
                     'BLK', 'BLK_opp', 'TOV', 'TOV_opp', 'PF',  'PF_opp', 'AdjEM']

    # Names of features that are not total season stats numbers
    not_divide_by_games = ['Seed', 'FG%', 'FG%_opp', '3P%', '3P%_opp', 'FT%', 'FT%_opp', 'AdjEM']

    # Divide Total Season Stats by the number of games to get per game stats
    for feature in feature_names:
        if feature not in not_divide_by_games:
            data['{}_Home'.format(feature)] = data['{}_Home'.format(feature)] / data['G_Home']
            data['{}_Away'.format(feature)] = data['{}_Away'.format(feature)] / data['G_Away']

    # Now that the data is in the correct format, create features
    return gen_features(data, feature_names)
