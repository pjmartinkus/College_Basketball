import os, six, urllib3, datetime
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


def load_scores_dataframe(start_date=None, end_date=None, csv_file_path=None):
    """
    Creates a csv of game scores for all games between and including the
    given start date and end date. The scores are retrieved from the sports
    reference website and then loaded into a csv file as a dataframe to be
    used for analysis.

    Args:
        start_date(datetime.date): Starting date to retrieve scores
        end_date(datetime.date): Ending date to retrieve scores
        csv_file_path(String): File path for the output .csv file

    Raises:
        AssertionError: If 'start_date' is not of type datetime.date.
        AssertionError: If 'end_date' is not of type datetime.date.
        AssertionError: If `csv_file_path` is not of type string.

    Example:
        >>> load_scores_dataframe()
    """

    # Check that the path is a string
    if csv_file_path is not None:
        if not isinstance(csv_file_path, six.string_types):
            raise AssertionError('Output file path must be a string.')

    # Load default values
    if start_date is None:
        start_date = datetime.date.today()
    if end_date is None:
        end_date = datetime.date.today()

    # Check that the dates make sense
    if type(start_date) is not datetime.date:
        raise AssertionError('Starting date must be of type datetime')
    if type(end_date) is not datetime.date:
        raise AssertionError('Ending date must be of type datetime')
    if start_date > end_date:
        raise AssertionError('The start date must be equal to or before the end date')

    # Check that files exits
    if csv_file_path is not None and os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    # Load the scores for each date in the date range
    current_date = start_date
    cols = ['Home', 'Home_Score', 'Away', 'Away_Score', 'Tournament']
    data_df = pd.DataFrame(columns=cols)
    while current_date <= end_date:

        # Create the correct url using the date
        url = 'https://www.sports-reference.com/cbb/boxscores/index.cgi?'
        url = url + 'month=' + current_date.strftime('%m')
        url = url + '&day=' + current_date.strftime('%d')
        url = url + '&year=' + current_date.strftime('%Y')

        # Get the webpage html
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        http = urllib3.PoolManager()
        r = http.request('get', url)

        # Parse the html document to get the table of data
        soup = BeautifulSoup(r.data, features='html.parser')

        # Iterate through each game
        data = []
        for game in soup.find_all('tbody'):

            # Get teams and score for each game
            vals = []
            tournament = np.nan
            for row in game.find_all('td'):
                if 'school' in str(row) or '<a>' in str(row):
                    if row.find('a') is not None:
                        vals.append(str(row.find('a').text))
                elif 'class="right"' in str(row):
                    vals.append(str(row.text))
                elif 'class="desc"' in str(row):
                    tournament = str(row.text)
            data.append(vals[0:4] + [tournament])

        # Create a dataframe for this day and add it to the previous days
        current_data = pd.DataFrame(data, columns=cols)
        data_df = pd.concat([data_df, current_data])

        # Increment the current date
        current_date = current_date + datetime.timedelta(days=1)

    # Rearrange the columns
    data_df = data_df[['Home', 'Away', 'Home_Score', 'Away_Score', 'Tournament']]

    if csv_file_path is not None:
        # write the final csv to a file
        data_df.to_csv(path_or_buf=csv_file_path, index=False)

    return data_df
