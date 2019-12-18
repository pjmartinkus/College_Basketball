import os, six, urllib3, datetime
from bs4 import BeautifulSoup
import pandas as pd


def load_stats_dataframe(year=None, csv_file_path=None):
    """
    Creates a pandas dataframe for each team in the input year
    for basic statistics found on the sports-reference website
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
        >>> load_stats_dataframe()
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

    # Get both offensive and defensive stats
    dataframes = []
    links = ['https://www.sports-reference.com/cbb/seasons/{}-opponent-stats.html'.format(year),
             'https://www.sports-reference.com/cbb/seasons/{}-school-stats.html'.format(year)]
    for link in links:

        # Get the webpage html
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        http = urllib3.PoolManager()
        r = http.request('get', link)

        # Parse the html document to get the table of data
        soup = BeautifulSoup(r.data, features='html.parser')

        # Get the column names from the header
        table_header = soup.find('thead').find_all('tr')[1]
        column_headers = [th.get_text() for th in table_header.find_all('th')]
        cols = [header for header in column_headers]
        cols.remove('Rk')
        cols.remove('\xa0')

        # Iterate through rows to get data
        data_array = []
        table_body = soup.find('tbody')
        rows = table_body.find_all('tr')
        for row in rows:

            # Check that this row is not a header row
            if len(row.find_all('th')) < 2:

                # Create list of values for this row
                vals = [value.text for value in row.find_all('td')]
                vals.remove('')
                data_array.append(vals)

        # Create a dataframe
        df = pd.DataFrame(data_array, columns=cols)
        df = df.drop(['W', 'L', 'W-L%', ], axis=1)

        # Fix school names issue and append to df list
        df['School'] = df['School'].str.replace('\xa0NCAA', '')
        dataframes.append(df)

    # Join offense and defense data into single dataframe
    on_cols = list(dataframes[0].columns)
    on_cols = on_cols[0:on_cols.index('FG')]
    data_df = dataframes[0].merge(dataframes[1], on=on_cols, suffixes=('_opp', ''))

    # Save the dataframe to csv if save_data is True
    if csv_file_path is not None:
        # Check if files exits
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)

        data_df.to_csv(csv_file_path, index=False)

    return data_df
