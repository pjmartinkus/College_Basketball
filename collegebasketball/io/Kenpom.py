import os, six, urllib3, datetime, re
import pandas as pd
from bs4 import BeautifulSoup


def load_kenpom_dataframe(year=None, csv_file_path=None):
    """
    Creates a pandas dataframe from the kenpom website to be used
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
        >>> load_kenpom_dataframe(year=2018, csv_file_path='path/to/file.csv')
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
    http = urllib3.PoolManager(cert_reqs='CERT_NONE')
    r = http.request('get', 'https://kenpom.com/index.php?y={}'.format(year))

    # Parse the html document to get the table of data
    soup = BeautifulSoup(r.data, features='html.parser')

    # Get the column names from the header
    table_header = soup.find('thead').find_all('tr')[1]
    column_headers = [th.get_text() for th in table_header.find_all('th')]
    cols = []
    for i, header in enumerate(column_headers):

        # Fix some column names
        if header == 'Rk':
            header = 'Rank'
        elif header == 'Team':
            cols.append('Team')
            header = 'Seed'
        elif header == 'W-L':
            cols.append('Wins')
            header = 'Losses'
        elif i > 11:
            header = 'NCSOS {}'.format(header)
        elif i > 8 and 'Opp' not in header:
            header = 'Opp{}'.format(header)

        # Add the header to the list
        cols.append(header)

        # Add the team rank columns
        if i > 4:
            cols.append('{} Rank'.format(header))

    # Iterate through rows to get data
    data_array = []
    table_body = soup.find('tbody')
    rows = table_body.find_all('tr')
    for row in rows:

        # Extract data for each row
        vals = []
        for value in row.find_all('td'):
            text = re.sub('[+]', '', value.text)
            vals.append(text)

        # Make sure this is not a header row
        if len(vals) > 5:

            # Split W-L into two values
            w_l = vals[3].split('-')
            vals[3] = w_l[0]
            vals.insert(4, w_l[1])

            # Find the tournament seed in the team name
            seed = re.findall(r'[0-9]+', vals[1])
            vals.insert(2, None)
            if len(seed) > 0:
                vals[2] = seed[0]

            # Remove tournament seed from team name
            vals[1] = re.sub(r'[0-9]+', '', vals[1]).strip()

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
