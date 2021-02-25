import os
import pandas as pd


def select_dataset(data, source='kenpom', drop=False, column_info_path=None):
    """
    Function to simplify the code to select or drop columns from a given data source/

    Args:
        data(DataFrame): A pandas DataFrame with feature vectors.
        source(String): The data source to keep or drop. Must be one of 'kenpom',
                        't-rank' or 'sports-reference'.
        drop(Boolean): Whether to drop the columns from the data. If False, then only
                       return columns from that data source and if True, then drop the
                       columns from that data source.
        column_info_path(String): Path to the columns per data source csv file. If None,
                                  will use the default value.

    Returns:
        The input DataFrame filtered for games in the specified tournament

    Raises:
        AssertionError: If data is not of type pandas DataFrame.
        AssertionError: If source not one of 'kenpom', 't-rank' or 'sports-reference'.
    """

    # Check that data is a dataframe
    if not isinstance(data, pd.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    # Check that source is one of our datasets
    possible_sources = ['kenpom', 't-rank', 'sports-reference']
    if source not in possible_sources:
        raise AssertionError('Input source must be one of {}'.format(possible_sources))

    # If a resource path not provided use the default
    if column_info_path is None:
        column_info_path = os.path.dirname(os.path.abspath(__file__)) + '/../../Data/Resources/dataset_columns.csv'

    # Get the list of columns for that data source and filter the df
    cols_df = pd.read_csv(column_info_path).set_index('source')
    source_cols = eval(cols_df.loc[source][0])
    if not drop:
        common_cols = eval(cols_df.loc['all'][0])
        source_cols = common_cols + source_cols
    select_cols = [col for col in data.columns if (col in source_cols) != drop]
    return data[select_cols]
