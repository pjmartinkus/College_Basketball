import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt


# Performs cross validation using scikit learn's cross validation function
def cross_val(data, exclude, models, model_names, scoring='f1', folds=5):
    """
    Performs k-fold cross validation using scikit learn's cross validation function
    and the input classifiers and data.

    Args:
        data(DataFrame): Input data to train and test the classifiers. This table must
                         include all of the features in the features list.
        exclude(list): List of columns to ignore during training and testing
        models(list): The list of classifiers to use during the training/testing.
        model_names(list): The names of models that will be included in the output table.
        scoring(String): The scoring parameter passed on to the Scikit learn CV function.
        folds(Int): The number of folds in the k-fold cross validation.

    Returns:
        A new pandas DataFrame that includes the scores for each model during the cross
        validation.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
    """

    # Check that data is a dataframes
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas DataFrame.')

    rows = []
    cols = ['Classifier']

    for i, model in enumerate(models):
        # Create the iterater
        cv = KFold(folds, shuffle=True, random_state=0)

        # Run the cross validation
        scores = cross_val_score(model, data.drop(exclude, axis=1), data['Label'], scoring=scoring, cv=cv)

        # Create a dataframe row to display the scores
        tuple = {}
        total = 0
        tuple['Classifier'] = model_names[i]
        for j, score in enumerate(scores):
            tuple['Run ' + str(j+1)] = score
            total += score

            # Create dataframe columns on first run
            if i == 0:
                cols.append('Run ' + str(j+1))
        tuple['Average'] = float(total) / len(scores)
        rows.append(tuple)

    cols.append('Average')
    return pd.DataFrame(rows, columns=cols)


def leave_march_out_cv(season, march, exclude, model):
    """
    Performs a custom type of cross validation using separated regular season and
    march data sets. The idea is to leave out one year's worth of march data as the
    test set and to use the rest of the data as the training set for each fold in the
    cross validation. The hope is that the results will better reflect the accuracy
    of each model when it is later actually used to predict the NCAA tournament games.

    Args:
        season(DataFrame): A pandas DataFrame of regular season data.
        march(DataFrame): A pandas DataFrame of march data.
        exclude(list): List of columns to ignore during training and testing
        model(Scikit-Learn Classifier): The model that will used to train/test.

    Returns:
        A new pandas DataFrame that includes the scores for each model during the cross
        validation.

    Raises:
        AssertionError: If season is not of type pandas DataFrame.
        AssertionError: If march is not of type pandas DataFrame.
    """

    # Check that season and march are dataframes
    if not isinstance(season, pd.DataFrame):
        raise AssertionError('Input argument season must be a pandas DataFrame.')
    if not isinstance(march, pd.DataFrame):
        raise AssertionError('Input argument march must be a pandas DataFrame.')

    # Get attributes used for the models
    features = list(season.columns)
    for col in exclude:
        features.remove(col)

    # Find out which years are in this data
    years = march.Year.unique()

    # Each fold leaves out a year of march data
    rows = []
    data_with_preds = []
    cols = ['Classifier', 'Precision', 'Recall', 'F1', 'AUC', 'Brier', 'Accuracy']
    for year in years:

        # Create our train and test data sets
        train = pd.concat([season, march[march['Year'] != year]])
        test = march[march['Year'] == year]

        # Train and run model
        model.fit(train[features], train[['Label']].values.ravel())
        predictions = model.predict(test[features])
        probabilities = model.predict_proba(test[features])
        data = test.copy()
        data['Prediction'] = predictions

        # Get probability given for class 1
        index1 = model.classes_.tolist().index(1)
        probabilities = [row[index1] for row in probabilities]
        data['Probability'] = probabilities

        # Get stats like precision, recall, ect.
        stats = get_stats(data, year)
        rows.append(stats)

        # Save predictions for later
        data_with_preds.append(data.loc[:, ['Favored', 'Underdog', 'Year', 'Label',
                                            'Prediction', 'Probability']])

    # Create data frame and add average row
    cv_results = pd.DataFrame(rows, columns=cols).set_index('Classifier').sort_index()
    cv_results.append(cv_results.mean().rename('Average'))

    return cv_results, pd.concat(data_with_preds)


def evaluate(train, test, exclude, models, model_names):
    """
    Computes the precision and recall of a model trained on the training data
    and tested on the test data.

    Args:
        train(DataFrame): A pandas DataFrame of training data.
        test(DataFrame): A pandas DataFrame of test data.
        exclude(list): List of columns to ignore during training and testing
        models(list): A List of the Scikit-Learn classifiers that will used to train/test.
        model_names(List): The names of models that will be included in the output table.

    Returns:
        A new pandas DataFrame that includes the scores for each model during evaluation.

    Raises:
        AssertionError: If the length of the model list is not equal to the length of the
                        model_names list.
        AssertionError: If train is not of type pandas DataFrame.
        AssertionError: If train is not of type pandas DataFrame.
    """

    # Check that season and march are dataframes
    if not isinstance(train, pd.DataFrame):
        raise AssertionError('Input argument season must be a pandas DataFrame.')
    if not isinstance(test, pd.DataFrame):
        raise AssertionError('Input argument march must be a pandas DataFrame.')

    # Confirm that length of models and model_names is the same
    if len(models) != len(model_names):
        raise AssertionError('Length of models list ({0}) does not equal ' +
                             'length of model names ({1}).'.format(len(models), len(model_names)))

    rows = []
    cols = ['Classifier', 'Precision', 'Recall', 'F1', 'AUC', 'Brier', 'Accuracy']
    features = list(train.columns)
    for col in exclude:
        features.remove(col)

    for i, model in enumerate(models):
        model.fit(train[features], train[['Label']].values.ravel())
        predictions = model.predict(test[features])
        probabilities = model.predict_proba(test[features])
        data = test.copy()
        data['Prediction'] = predictions
        data['Probability'] = probabilities[:, 1]

        stats = get_stats(data, model_names[i])
        rows.append(stats)

    return pd.DataFrame(rows, columns=cols)


def probability_graph(actual, probabilities, model_names=None, start=0.0, stop=0.7, bin_width=0.05, **kwargs):
    """
    Creates a line plot visualization plotting the predicted probabilities from a model
    against the fraction of actual upsets for those predictions. The functions splits
    the predicted probabilities into bins spanning from the given starting and stopping
    values. Then the fraction of upsets for the games in each bin is calculated and plotted
    in a line plot. If the `probabilities` argument is a list of lists, the function
    will calculate those upset percentages for each list of probabilities and plot them
    in multiple series.

    The goal is to provide a visualization to help determine if the probability given by
    the classifier is actually representative of the probability the classifier is correct.

    Args:
        actual(List): A list like object of the correct result for a set of feature vectors.
        probabilities(List): A list (or list of lists if plotting multiple series) of predicted
                             probabilities for the same games as the `actual` input list.
        model_names(List): The names of each series of probabilities if plotting multiple series
        start(Float): The start of the range of probabilities to cover. Should be between 0 and 1.
        stop(Float): The end of the range of probabilities to cover. Should be between 0 and 1.
        bin_width(Float): The width of each bin.
        kwargs: Key word arguments to pass on to the figure functions from matplotlib.
    """
    num_bins = int((stop - start) / bin_width) + 1
    upset_pcts = list()
    for probs in probabilities:
        actual_and_probs = list(zip(actual, probs))

        # Bin predictions (bins from start to end with widths of bin_width)
        this_model_upset_pcts = list()
        for i in range(num_bins):
            # Get upper and lower bound for bins
            bin_range = [round(start + bin_width * i, 3), round(start + bin_width * (i + 1), 3)]

            # Calculate percentage of upsets for games with predicted probabilities within the bin
            bin_games = [label for label, prob in actual_and_probs if bin_range[0] < prob <= bin_range[1]]
            bin_pct_upsets = sum(bin_games) / len(bin_games)
            this_model_upset_pcts.append(bin_pct_upsets)
        upset_pcts.append(this_model_upset_pcts)

    # Plot results
    bin_midpoints = [round(bin_width / 2 + i * bin_width, 3) for i in range(num_bins)]
    fig = plt.figure(**kwargs)
    ax = plt.axes()

    for x, name in zip(upset_pcts, model_names):
        ax.plot(bin_midpoints, x, label=name)

    ax.legend()
    plt.ylabel('Fraction of Games that were Upsets')
    plt.xlabel('Predicted Probability of Upset')


# Calculate the precision, recall and f1 score for the input data.
def get_stats(data, model_name):
    precision = precision_score(data['Label'], data['Prediction'])
    recall = recall_score(data['Label'], data['Prediction'])
    f1 = f1_score(data['Label'], data['Prediction'])
    auc = roc_auc_score(data['Label'], data['Probability'])
    brier = brier_score_loss(data['Label'], data['Probability'])
    accuracy = accuracy_score(data['Label'], data['Prediction'])

    return [model_name, precision, recall, f1, auc, brier, accuracy]


def post_tournament_eval(predictions, scores, kenpom):
    """
    Calculates some metrics once the NCAA Tournament is over to evaluate how the model
    performed. The metrics include the accuracy of the model over each round of the
    tournament.

    Args:
        predictions(DataFrame): A pandas DataFrame of predictions.
        scores(DataFrame): A pandas DataFrame of game score data from the tournament.
        kenpom(DataFrame): A pandas DataFrame of Kenpom data used to determine who was favored
                      in each game.

    Returns:
        A new pandas DataFrame that includes accuracy metrics for each round.

    Raises:
        AssertionError: If any of the input dataframes are not of type pandas DataFrame.
    """

    # Check that input arguments are actually dataframes
    if not isinstance(predictions, pd.DataFrame):
        raise AssertionError('Input argument predictions must be a pandas DataFrame.')
    if not isinstance(scores, pd.DataFrame):
        raise AssertionError('Input argument scores must be a pandas DataFrame.')
    if not isinstance(kenpom, pd.DataFrame):
        raise AssertionError('Input argument kenpom must be a pandas DataFrame.')

    # Join kenpom and scores
    data = pd.merge(scores, kenpom[['Team', 'Rank']], left_on='Home', right_on='Team', sort=False)
    data = pd.merge(data, kenpom[['Team', 'Rank']], left_on='Away', right_on='Team', suffixes=('_Home', '_Away'),
                    sort=False)

    # Split data by round
    round_names = ['First Round', 'Second', 'Regional Semifinal', 'Regional Final', 'National Semifinal',
                   'National Final']
    round_idx = [(0, 32), (32, 48), (48, 56), (56, 60), (60, 62), (62, 63)]

    results = list()
    for (n, i) in zip(round_names, round_idx):
        preds = predictions[i[0]:i[1]]
        games = data[data['Tournament'].apply(lambda x: n in x)]

        # Create some intermediate data sets
        all_teams = pd.concat([games['Home'], games['Away']])
        all_predicted_teams = pd.concat([preds['Favored'], preds['Underdog']])
        winners = pd.concat([games[games['Home_Score'] > games['Away_Score']]['Home'],
                             games[games['Home_Score'] < games['Away_Score']]['Away']])
        underdog = pd.concat([games[games['Rank_Home'] > games['Rank_Away']]['Home'],
                              games[games['Rank_Home'] < games['Rank_Away']]['Away']])

        # Calculate Metrics
        has_pred = preds[preds['Predicted Winner'].isin(all_teams)]
        correct = preds[preds['Predicted Winner'].isin(winners)]
        upsets = underdog[underdog.isin(winners)]
        upsets_predicted = preds[preds['Predicted Winner'] == preds['Underdog']]
        has_pred_upset = upsets_predicted[upsets_predicted['Predicted Winner'].isin(all_teams)]
        has_actual_upset = upsets[upsets.isin(all_predicted_teams)]
        correct_upsets = correct[correct['Predicted Winner'].isin(underdog)]
        correct_predicted_upsets = upsets_predicted[upsets_predicted['Predicted Winner'].isin(winners)]

        results.append([len(games), len(has_pred), len(correct), len(upsets),
                        len(upsets_predicted), len(has_actual_upset),
                        len(has_pred_upset), len(correct_upsets), len(correct_predicted_upsets)])

    # Add in total row
    cols = ['Games', 'Games Contained Predicted Winner', 'Correct Predictions', 'Total Upsets',
            'Upsets Predicted', 'Games Containing Actual Upset Winner',
            'Games Containing Predicted Upset Winner', 'Correct Upsets', 'Correct Predicted Upsets']
    results = pd.DataFrame(results, columns=cols)
    results = pd.concat([results, pd.DataFrame(results.sum()).transpose()])

    # Add in metric columns
    results['Total Accuracy'] = results['Correct Predictions'] / results['Games']
    results['Upset Precision'] = results['Correct Predicted Upsets'] / results['Upsets Predicted']
    results['Upset Recall'] = results['Correct Upsets'] / results['Total Upsets']
    results['Adj Accuracy'] = results['Correct Predictions'] / results['Games Contained Predicted Winner']
    results['Adj Upset Precision'] = results['Correct Predicted Upsets'] / \
                                     results['Games Containing Predicted Upset Winner']
    results['Adj Upset Recall'] = results['Correct Upsets'] / results['Games Containing Actual Upset Winner']
    return results
