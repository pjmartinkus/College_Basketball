from collegebasketball.features.Features import feature_difference, win_loss, teams
import pandas as pd


def gen_features(data):

    feature_vecs = []
    feature_names = ['Rank', 'AdjEM', 'AdjO', 'AdjO Rank', 'AdjD', 'AdjD Rank', 'AdjT',
                     'AdjT Rank', 'Luck', 'Luck Rank', 'OppAdjEM', 'OppAdjEM Rank', 'OppO',
                     'OppO Rank', 'OppD', 'OppD Rank', 'NCSOS AdjEM', 'NCSOS AdjEM Rank']

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

        # Add the label if necessary
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
    return pd.DataFrame(feature_vecs, columns=cols), features
