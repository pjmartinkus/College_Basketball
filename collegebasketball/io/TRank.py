import os, six, urllib3, datetime
from bs4 import BeautifulSoup
import pandas as pd


def load_TRank_dataframe(year=None, csv_file_path=None):
    """
    Creates a pandas dataframe from the T-Rank website to be used
    for analysis. Can also create and save a csv file from the
    dataframe.

    Args:
        year(int): The year to get stats from
        csv_file_path(String): File path for the output .csv file.
                               If None, then no csv file is saved.

    Raises:
        AssertionError: If 'year' is not of type integer.
        AssertionError: If `csv_file_path` is not of type string.

    Example:
        >>> load_TRank_dataframe()
    """

    # Check that the path is a string
    if csv_file_path is not None:
        if not isinstance(csv_file_path, six.string_types):
            raise AssertionError('Output file path must be a string.')

    # Check that the year is an int
    if year is not None:
        if not isinstance(year, int):
            raise AssertionError('Year must be an integer.')
    # Load default values if none given
    else:
        year = datetime.datetime.now().year

    # Get the webpage html
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    http = urllib3.PoolManager()
    r = http.request('get', 'http://barttorvik.com/trank.php?year={}&sort=&top=0&conlimit=All#'.format(year))

    # Parse the html document to get the table of data
    soup = BeautifulSoup(r.data, features='html.parser')

    # Get the column names from the header
    table_header = soup.find('thead').find_all('tr')[1]
    column_headers = [th.get_text() for th in table_header.find_all('th')]
    cols = []
    for i, header in enumerate(column_headers):
        cols.append(header)
        if header == 'Rec':
            cols[-1] = 'Wins'
            cols.append('Losses')
        elif i > 4:
            cols.append("{} Rank".format(header))

    # Iterate through rows to get data
    data_array = []
    table_body = soup.find('tbody')
    rows = table_body.find_all('tr')
    for row in rows:
        vals = []

        # Check that this row is not a header row
        if row.find('th') is None:
            for value in row.find_all('td'):
                text = value.text

                # Split into actual stat and rank
                if value.find('span') is not None:
                    rank = value.find('span').text
                    stat = text
                    if len(rank) > 0:
                        stat = stat[0:-len(rank)]

                    # Append the relevant information for this stat
                    if 'team=' not in str(value):
                        vals.extend([stat, rank])

                    # If this value is team record, split into W-L
                    elif len(vals) == 4:
                        stat = stat.split('-')
                        vals.append(stat[0])
                        vals.append(stat[1])

                    else:
                        vals.append(stat)

                else:
                    vals.append(text)

            data_array.append(vals)

    # Create a dataframe
    data_df = pd.DataFrame(data_array, columns=cols)

    # Save the dataframe to csv if save_data is True
    if csv_file_path is not None:
        # Check if files exits
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        data_df.to_csv(csv_file_path, index=False)

    return data_df
