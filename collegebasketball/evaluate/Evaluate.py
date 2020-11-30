import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt


# Performs cross validation using scikit learn's cross validation function
def cross_val(data, features, models, model_names, scoring='f1', folds=5):

    rows = []
    cols = ['Classifier']

    for i, model in enumerate(models):
        # Create the iterater
        cv = KFold(folds, shuffle=True, random_state=0)

        # Run the cross validation
        scores = cross_val_score(model, data[features], data['Label'], scoring=scoring, cv=cv)

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

    # Get attributes used for the models
    features = list(season.columns)
    for col in exclude:
        features.remove(col)

    # Find out which years are in this data
    years = march.Year.unique()

    # Each fold leaves out a year of march data
    rows = []
    data_with_preds = []
    cols = ['Classifier', 'Precision', 'Recall', 'F1', 'Accuracy']
    for year in years:

        # Create our train and test data sets
        train = pd.concat([season, march[march['Year'] != year]])
        test = march[march['Year'] == year]

        # Train and run model
        model.fit(train[features], train[['Label']])
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

    return pd.DataFrame(rows, columns=cols), pd.concat(data_with_preds)


# Computes the precision and recall of a model given the predicted and labeled data
def evaluate(train, test, exclude, models, model_names):

    # confirm that length of models and model_names is the same
    if len(models) != len(model_names):
        raise AssertionError('Length of models list ({0}) does not equal ' +
                             'length of model names ({1}).'.format(len(models), len(model_names)))

    rows = []
    cols = ['Classifier', 'Precision', 'Recall', 'F1', 'Accuracy']
    features = list(train.columns)
    for col in exclude:
        features.remove(col)

    for i, model in enumerate(models):
        model.fit(train[features], train[['Label']])
        predictions = model.predict(test[features])
        data = test.copy()
        data['Prediction'] = predictions

        stats = get_stats(data, model_names[i])
        rows.append(stats)

    return pd.DataFrame(rows, columns=cols)


def probability_hist(data, num_bins, start=0.4, stop=0.6, stat='f1'):

    diff = stop - start
    bin_midpoints = []
    bins = []
    for i in range(0, num_bins):
        # Get bin ranges and midpoints
        bin_range = [start + diff * i / num_bins, start + diff * (i + 1) / num_bins]
        bin_midpoints.append(sum(bin_range) / 2)

        # Get data in this bin
        bin_curr = data[data['Probability'] > bin_range[0]]
        bin_curr = bin_curr[bin_curr['Probability'] <= bin_range[1]]

        # Calculate the accuracy and precision in each bin
        stats = get_stats(bin_curr, None)
        ind = [None, 'precision', 'recall', 'f1', 'accuracy']
        bins.append(stats[ind.index(stat)])

        # Plot results
        plt.bar(bin_midpoints, bins, width=bin_range[1] - bin_range[0] - .001)


def get_stats(data, model_name):

    total_pos = len(data[data['Label'] == 1])
    total_pred_pos = len(data[data['Prediction'] == 1])

    predictions = data[data['Label'] == 1]
    true_pos = len(predictions[predictions['Prediction'] == 1])

    neg_preds = data[data['Label'] == 0]
    true_neg = len(neg_preds[neg_preds['Prediction'] == 0])

    if total_pred_pos > 0:
        precision = true_pos / float(total_pred_pos)
    else:
        precision = float('nan')

    if total_pos > 0:
        recall = true_pos / float(total_pos)
    else:
        recall = float('nan')

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = float('nan')

    if true_neg > 0:
        accuracy = (true_neg + true_pos) / float(len(data))
    else:
        accuracy = float('nan')

    return [model_name, precision, recall, f1, accuracy]
