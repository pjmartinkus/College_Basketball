import six, os, pandas


def load_csv(file_path):
    """
    Loads a csv file and returns it as a pandas dataframe

    Args:
        file_path(String): File path for the input .csv file

    Raises:
        AssertionError: If `file_path` is not of type string.
        AssertionError: If a file does not exist in the
            given `file_path`.

    Example:
        >>> load_csv('path/to/file')
    """

    # Check that paths are both strings
    if not isinstance(file_path, six.string_types):
        raise AssertionError('Input file path must be a string.')

    # Check that fies exits
    if not os.path.exists(file_path):
        raise AssertionError('File does not exist at path %s' % file_path)

    return pandas.read_csv(file_path, header=0)