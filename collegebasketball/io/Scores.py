import os, six, pandas, urllib3, datetime
from collegebasketball.io.ioHelper import load_csv


def load_scores_dataframe(html_file_path=None, csv_file_path=None, start_date=None, end_date=None, save_data=False):
    """
    Creates a csv of game scores for all games between and including the
    given start date and end date. The scores are retrieved from html files
    from sports reference and then loaded into a csv file as a dataframe
    to be used for analysis.

    Args:
        html_file_path(String): File path for the input .html file
        csv_file_path(String): File path for the output .csv file
        start_date(datetime.date): Starting date to retrieve scores
        end_date(datetime.date): Ending date to retrieve scores
        save_data(Boolean): If True, then the .csv file will be saved.

    Raises:
        AssertionError: If `html_file_path` is not of type string.
        AssertionError: If `csv_file_path` is not of type string.
        AssertionError: If 'start_date' is not of type datetime.date.
        AssertionError: If 'end_date' is not of type datetime.date.
        AssertionError: If 'start_date' is after 'end_date'.

    Example:
        >>> load_scores_dataframe()
    """

    # Check that paths are both strings
    if html_file_path is not None:
        if not isinstance(html_file_path, six.string_types):
            raise AssertionError('Input file path must be a string.')
    if csv_file_path is not None:
        if not isinstance(csv_file_path, six.string_types):
            raise AssertionError('Output file path must be a string.')

    # Load default values
    if html_file_path is None:
        html_file_path = '/Users/phil/Documents/Documents/College Basketball/Data/scores.html'
    if csv_file_path is None:
        csv_file_path = '/Users/phil/Documents/Documents/College Basketball/Data/scores.csv'
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
    if os.path.exists(html_file_path):
        os.remove(html_file_path)
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    # Load the scores for each date in the date range
    current_date = start_date
    dataframe = pandas.DataFrame()

    while current_date <= end_date:
        # Call functions to create pandas dataframe
        get_scores_html(html_file_path, current_date)
        scores_to_csv(html_file_path, csv_file_path)
        dataframe = pandas.concat([dataframe, load_csv(csv_file_path)])

        # Delete the created files
        os.remove(html_file_path)
        os.remove(csv_file_path)

        # Increment the current date
        current_date = current_date + datetime.timedelta(days=1)

    if save_data:
        # write the final csv to a file
        dataframe.to_csv(path_or_buf=csv_file_path, index=False)

    return dataframe


def get_scores_html(file_path=None, date=None):
    """
    Creates an html file from the sports reference webpage for the given date.

    Args:
        file_path(String) File path for the output .html file
        date(datetime.date) Retrieves scores html from games on this date

    Raises:
        AssertionError: If `file_path` is not of type string.
        AssertionError: If 'date' is not of type datetime.date.

    Example:
        >>> # Get today's scores html page
        >>> get_scores_html()
        >>> # Get scores html page from 01/20/17
        >>> score_date = datetime.date(2017, 1, 20)
        >>> get_scores_html(date=score_date)

    """

    # Check that path is a string
    if file_path is not None:
        if not isinstance(file_path, six.string_types):
            raise AssertionError('Output file path must be a string.')

    # Check that date is of type datetime
    if date is not None:
        if type(date) is not datetime.date:
            raise AssertionError('Argument date must be of type datetime.date')

    # Load default values
    if file_path is None:
        file_path = '/Users/phil/Documents/Documents/College Basketball/Data/scores.html'
    if date is None:
        date = datetime.date.today()

    # Check that file exits
    if os.path.exists(file_path):
        os.remove(file_path)

    # Create the correct url using the date
    url = 'https://www.sports-reference.com/cbb/boxscores/index.cgi?'
    url = url + 'month=' + date.strftime('%m')
    url = url + '&day=' + date.strftime('%d')
    url = url + '&year=' + date.strftime('%Y')

    # Get the webpage html
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    http = urllib3.PoolManager()
    r = http.request('get', url)

    with open(file_path, 'w') as fid:
        fid.write(r.data)


def scores_to_csv(input_file_path=None, output_file_path=None):
    """
    Creates a CSV file with data for game scores from an html page of scores from sports reference.

    Args:
        input_file_path(String): File path for the input .html file
        output_file_path(String): File path for the output .csv file

    Raises:
        AssertionError: If `input_file_path` is not of type string.
        AssertionError: If `output_file_path` is not of type string.
        AssertionError: If a file does not exist in the
            given `input_file_path`.

    Example:
        >>> scores_to_csv()
    """

    # Check that paths are both strings
    if input_file_path is not None:
        if not isinstance(input_file_path, six.string_types):
            raise AssertionError('Input file path must be a string.')
    if output_file_path is not None:
        if not isinstance(output_file_path, six.string_types):
            raise AssertionError('Output file path must be a string.')

    # Load default values
    if input_file_path is None:
        input_file_path = '/Users/phil/Documents/Documents/College Basketball/Data/scores.html'
    if output_file_path is None:
        output_file_path = '/Users/phil/Documents/Documents/College Basketball/Data/scores.csv'

    # Check that fies exits
    if not os.path.exists(input_file_path):
        raise AssertionError('File does not exist at path %s' % input_file_path)
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # open scores html file and csv file
    scores_html = open(input_file_path, 'r')
    scores_csv = open(output_file_path, 'w')

    scores_csv.write('Home,Away,Home_Score,Away_Score\n')

    # Keep track of when a line will contain a score
    score_line = False
    list = []

    # Go through each line in the html file to get scores
    for line in scores_html:

        # Write the items to csv file when we have all the information
        if len(list) == 4:

            # Write the new line to the file
            scores_csv.write(','.join(list) + '\n')

            # Reset the list
            list = []

        # If this line contains a score
        if score_line:
            # Get the numeric characters from the line
            score = ''.join(ch for ch in line if ch.isdigit())

            # Add the score to the list
            list.append(score)

            # Reset score_line
            score_line = False

        # We only care about lines for each team
        if 'cbb/schools' in line and '<td>' in line:

            # Remember that the next line will contain a score
            score_line = True

            # Keep track of indexes
            start = 0
            i = 0

            # Go through each character in the string
            for c in line:
                # When we see a '>' remember the index
                if c == '>':
                    start = i

                # When we see a '<' save the string
                if c == '<':
                    # Save the substring and remove any '+' characters
                    item = line[start + 1:i].replace('+', '')

                    # Check if the item contains actual info save to list
                    if item != '' and item != ' ' and '\t' not in item and 'nbsp' not in item:
                        # If the list is empty, then we don't have the home team
                        if len(list) == 0:
                            list.append(line[start + 1:i])
                        else:
                            list.insert(1, line[start + 1:i])

                i = i + 1
