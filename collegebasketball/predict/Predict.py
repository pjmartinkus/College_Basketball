from collegebasketball.features.FeatureGen import gen_features
import pandas as pd


def predict(model, games, features):
    """
    Generates predictions for the given NCAA Tournament using a list of the games
    for the first round. The games DataFrame should follow the same format as the
    training data used to train the input classifier. The games should be ordered such
    that the winner of games in consecutive rows will play in the next round. This
    function will then generate predictions for each of the games and then continue
    making predictions for each of the later rounds until the champion is predicted.


    Args:
        model(Scikit-Learn Classifier): A trained Scikit-Learn classifier.
        games(DataFrame): A DataFrame containing all of the matchups in the first
                          round of the tournament.
        features(list): A list of the features that will be used to generate
                        predictions with the classifier.

    Returns:
        A pandas DataFrame that includes the names of the favored team, underdog,
        predicted winner and the probability returned by the classifier for each
        game in the tournament for the input data.

    Raises:
        AssertionError: If games is not of type pandas DataFrame.
    """

    # Check that games is a dataframes
    if not isinstance(games, pd.DataFrame):
        raise AssertionError('Input games argument must be a pandas DataFrame.')

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
        features_to_gen = [x for x in features if '_Fav' not in x and x != 'Win_Loss']
        data = gen_features(games, features_to_gen)

        # Set the threshold probability for an upset for each game
        if i == 0:  # First round we'll consider the difference in seeding
            thresholds = 0.511 - abs(games['Seed_Home'] - games['Seed_Away']) * 0.03
        elif i < 2:  # Second round we'll allow more upsets with threshold of 0.3
            thresholds = [0.3 for _ in range(len(games))]
        elif i < 3:  # Middle rounds we'll allow slightly fewer with threshold of 0.32
            thresholds = [0.32 for _ in range(len(games))]
        else:  # Later rounds we'll tighten up the threshold to 0.45 to get a more accurate champion
            thresholds = [0.45 for _ in range(len(games))]

        # Get predictions for the this round
        probs = model.predict_proba(data[features])
        data['Probabilities'] = [prob[1] for prob in probs]
        predictions = []
        for prob, threshold in zip(probs, thresholds):
            if prob[1] > threshold:
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

            # Show the winner
            if row[data.columns.get_loc('Prediction')] == 0:
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
