from collegebasketball.features.Features import feature_difference, win_loss, teams
import pandas as pd


def gen_features(data, feature_names):

    feature_vecs = []

    # Create the column names list
    cols = ['Favored', 'Underdog', 'Year', 'Win_Loss_Fav', 'Win_Loss', 'Win_Loss_Diff']
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
        vec = {}
        vec['Favored'], vec['Underdog'] = teams(row, columns)
        vec['Year'] = row[columns.get_loc('Year')]

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

    # Create a list of all the features used
    features = list(cols)
    remove = ['Favored', 'Underdog', 'Label', 'Year']
    for feat in remove:
        if feat in features:
            features.remove(feat)

    # Return the dataframe
    return pd.DataFrame(feature_vecs, columns=cols)


def gen_kenpom_features(data):

    # Names of all the features for Kenpom
    feature_names = ['Rank', 'AdjEM', 'AdjO', 'AdjO Rank', 'AdjD', 'AdjD Rank', 'AdjT',
                     'AdjT Rank', 'Luck', 'Luck Rank', 'OppAdjEM', 'OppAdjEM Rank', 'OppO',
                     'OppO Rank', 'OppD', 'OppD Rank', 'NCSOS AdjEM', 'NCSOS AdjEM Rank']

    return gen_features(data, feature_names)


def gen_TRank_features(data, kenpom_data):

    # Names of the features for T-Rank
    feature_names = ['Rk', 'AdjOE', 'AdjOE Rank', 'AdjDE', 'AdjDE Rank', 'Barthag',
                     'EFG%', 'EFG% Rank', 'EFGD%', 'EFGD% Rank', 'TOR', 'TOR Rank',
                     'TORD', 'TORD Rank', 'ORB', 'ORB Rank', 'DRB', 'DRB Rank',
                     'FTR', 'FTR Rank', 'FTRD', 'FTRD Rank', '2P%', '2P% Rank',
                     '2P%D', '2P%D Rank', '3P%D', '3P%D Rank', 'Adj T.', 'Adj T. Rank',
                     'WAB', 'WAB Rank']

    # Add the win/loss columns from kenpom
    data = data.assign(Wins_Home=kenpom_data['Wins_Home'], Losses_Home=kenpom_data['Losses_Home'],
                       Wins_Away=kenpom_data['Wins_Away'], Losses_Away=kenpom_data['Losses_Away'])

    # Add Kenpom Rankings columns for home and away team to tell which team was favored by kenpom
    data = data.assign(AdjEM_Home=kenpom_data['AdjEM_Home'], AdjEM_Away=kenpom_data['AdjEM_Away'])

    # Now that the data is in the correct format, create features
    return gen_features(data, feature_names)


def gen_basic_features(data, kenpom_data):

    # Names of the features for the basic stats
    feature_names = ['Tm.', 'Opp.', 'MP', 'FG', 'FG_opp', 'FGA', 'FGA_opp', 'FG%',
                     'FG%_opp', '3P', '3P_opp', '3PA', '3PA_opp', '3P%', '3P%_opp',
                     'FT', 'FT_opp', 'FTA', 'FTA_opp', 'FT%', 'FT%_opp', 'ORB',
                     'ORB_opp', 'TRB', 'TRB_opp', 'AST', 'AST_opp', 'STL', 'STL_opp',
                     'BLK', 'BLK_opp', 'TOV', 'TOV_opp', 'PF',  'PF_opp']

    # Add Kenpom Rankings columns for home and away team to tell which team was favored by kenpom
    data = data.assign(AdjEM_Home=kenpom_data['AdjEM_Home'], AdjEM_Away=kenpom_data['AdjEM_Away'])

    # Add the win/loss columns from kenpom
    data = data.assign(Wins_Home=kenpom_data['Wins_Home'], Losses_Home=kenpom_data['Losses_Home'],
                       Wins_Away=kenpom_data['Wins_Away'], Losses_Away=kenpom_data['Losses_Away'])

    # Divide Total Season Stats by the number of games to get per game stats
    for feature in feature_names:
        if '%' not in feature:
            data['{}_Home'.format(feature)] = data['{}_Home'.format(feature)] / data['G_Home']
            data['{}_Away'.format(feature)] = data['{}_Away'.format(feature)] / data['G_Away']

    # Now that the data is in the correct format, create features
    return gen_features(data, feature_names)
