import pandas as pd


def block_table(data):
    """
    Removes all tuples from the input DataFrame that are blocked by out blocking scheme.

    Args:
        data(DataFrame): Input data to block.

    Returns:
        A pandas DataFrame that includes all tuples from the input DataFrame that were
        not blocked by the blocking scheme.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas DataFrame.')

    rows = []
    cols = data.columns

    # We want to apply the rules for each tuple
    for row in data.itertuples(index=False):

        # Check if the tuple will be blocked
        block = blocking_rules(row, cols)

        # If we decide not to block the tuple, keep it in the new table
        if not block:
            rows.append(row)

    # Return the pruned dataframe
    cols = data.columns
    return pd.DataFrame(rows, columns=cols)


def debug(data):
    """
    Returns all true games that were blocked by our blocking scheme. This can be
    a useful tool to make sure that the blocking scheme is not too aggressive.

    Args:
        data(DataFrame): Input data to test the blocking scheme with.

    Returns:
        A pandas DataFrame that includes all positive examples tuples from the
        input DataFrame that were blocked by the blocking scheme.

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas DataFrame.')

    rows = []
    cols = data.columns

    # We want to apply the rules for each tuple
    for row in data.itertuples(index=False):

        # Check if the tuple will be blocked
        block = blocking_rules(row, cols)

        # If we decide to block but the label is 1, add to list
        if block and row[cols.get_loc('Label')] == 1:
            rows.append(row)

    # Return the pruned dataframe
    return pd.DataFrame(rows, columns=cols)


# Determines if the tuples will be blocked or not
def blocking_rules(row, cols):

    block = False

    # If one team is obviously superior
    if row[cols.get_loc('AdjEM_Diff')] > 30:
        block = True

    # If either team is not a tournament team
    if row[cols.get_loc('AdjEM')] < -12 or row[cols.get_loc('AdjEM_Fav')] < -5:
        block = True

    return block


# Experimental function designed to help find blocking rules.
def create_rule(data, feat):

    # Keep track of the rules
    rules = []
    cols = ['Rule', 'Rows_Blocked', 'Correct_Blocked']

    # Create the step size
    steps = 20
    min = data[feat].min()
    max = data[feat].max()
    step_size = (max - min) / steps

    # Try a rule for each step
    total = len(data)
    total_upsets = len(data[data['Label'] == 1])
    for i in range(steps):
        split = min + i * step_size
        rows = []

        # Try a rule in the form feature > split
        correct_blocked = 0
        rows_blocked = 0
        for row in data.itertuples(index=False):
            # Check if the tuple will be blocked
            if row[data.columns.get_loc(feat)] > split:
                rows_blocked += 1
                # Check if the row shouldn't have been blocked
                if row[data.columns.get_loc('Label')] == 1:
                    correct_blocked += 1

        # Evaluate the rule
        correct_blocked /= float(total_upsets)
        rows_blocked /= float(total)

        # Add the rule
        if correct_blocked < 0.1 < rows_blocked:
            rules.append([feat + '_gt_' + str(split), rows_blocked, correct_blocked])

        # Try a rule in the form feature < split
        correct_blocked = 0
        rows_blocked = 0
        for row in data.itertuples(index=False):
            # Check if the tuple will be blocked
            if row[data.columns.get_loc(feat)] < split:
                rows_blocked += 1
                # Check if the row shouldn't have been blocked
                if row[data.columns.get_loc('Label')] == 1:
                    correct_blocked += 1

        # Evaluate the rule
        correct_blocked /= float(total_upsets)
        rows_blocked /= float(total)

        # Add the rule
        if correct_blocked < 0.1 < rows_blocked:
            rules.append([feat + '_lt_' + str(split), rows_blocked, correct_blocked])

    return pd.DataFrame(rules, columns=cols)
