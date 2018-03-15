import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import time


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


# Computes the precision and recall of a model given the predicted and labeled data
def evaluate(train, test, features, models, model_names):

    rows = []
    cols = ['Classifier', 'Precision', 'Recall', 'F1', 'Accuracy']

    for i, model in enumerate(models):
        model.fit(train[features], train[['Label']])
        predictions = model.predict(test[features])
        data = test.copy()
        data['Prediction'] = predictions

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

        rows.append([model_names[i], precision, recall, f1, accuracy])

    return pd.DataFrame(rows, columns=cols)
