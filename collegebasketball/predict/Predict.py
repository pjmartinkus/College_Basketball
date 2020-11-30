from collegebasketball.features.FeatureGen import gen_features
from collegebasketball.blocking.Blocker import block_table
import pandas as pd


def predict(model, games):
    preds = []
    cols = games.columns
    pred_cols = ['Favored', 'Underdog', 'Predicted Winner', 'Probabilities']

    # Get the features for the home and away teams
    home = ['Home']
    away = ['Away']
    for feat in games.columns:
        if 'Home' in feat:
            home.append(feat)
        elif 'Away' in feat:
            away.append(feat)

    # Generate predictions for each round
    for i in range(6):
        next_round = []

        # Generate the Features
        data, features = gen_features(games)

        # Block the tables
        blocked = data

        # Get predictions for the this round
        probs = model.predict_proba(data[features])
        data['Probabilities'] = [prob[1] for prob in probs]
        predictions = []
        for prob in probs:
            if prob[1] > 0.4985:
                predictions.append(1)
            else:
                predictions.append(0)
        data['Prediction'] = predictions

        j = 0
        for row in data.itertuples(index=False):

            # Simplify the data for presentation
            tuple = {}
            tuple['Favored'] = row[data.columns.get_loc('Favored')]
            tuple['Underdog'] = row[data.columns.get_loc('Underdog')]
            tuple['Probabilities'] = row[data.columns.get_loc('Probabilities')]

            # Check if this tuple would have been blocked
            if len(blocked.loc[blocked['Favored'] == tuple['Favored'], :]) == 0:
                tuple['Predicted Winner'] = tuple['Favored']

            # Show the winner
            elif row[data.columns.get_loc('Prediction')] == 0:
                tuple['Predicted Winner'] = tuple['Favored']
            else:
                tuple['Predicted Winner'] = tuple['Underdog']
            preds.append(tuple)

            # Find out if winner was home or away
            home_win = False
            if games.iloc[j, cols.get_loc('Home')] == tuple['Predicted Winner']:
                home_win = True

            # Get data for home in next game
            if j % 2 == 0:
                next_tuple = {}
                for k, feat in enumerate(home):
                    if home_win:
                        next_tuple[feat] = games.iloc[j, cols.get_loc(feat)]
                    else:
                        next_tuple[feat] = games.iloc[j, cols.get_loc(away[k])]
            else:
                for k, feat in enumerate(away):
                    if home_win:
                        next_tuple[feat] = games.iloc[j, cols.get_loc(home[k])]
                    else:
                        next_tuple[feat] = games.iloc[j, cols.get_loc(feat)]
                next_round.append(next_tuple)
            j += 1
        games = pd.DataFrame(next_round, columns=cols)

    return pd.DataFrame(preds, columns=pred_cols)