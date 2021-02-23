import pandas as pd


def filter_tournament(data, col='Tournament', tourney='NCAA', drop=False):
    """
    Function to simplify the code to filter a data set for games from a specific tournament.

    Args:
        data(DataFrame): A pandas DataFrame with the game scores data.
        col(String): The column containing the tournament information.
        tourney(String): The tournament to filter games for.
        drop(Boolean): Whether to drop the games from the tournament. If False,
                       then only return games from that tournament and if True, then
                       return only games that were not in that tournament.

    Returns:
        The input DataFrame filtered for games in the specified tournament

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
        AssertionError: If col in not a column in the input DataFrame.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    # Check that col is an actual column in the dataframe
    if not col in data.columns:
        raise AssertionError('Input col must be a column in the input dataframe')

    return data[data[col].str.contains(tourney, na=False) != drop]
