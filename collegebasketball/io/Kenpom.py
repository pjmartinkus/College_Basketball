import os, six, urllib3, datetime
from collegebasketball.io.ioHelper import load_csv


def load_kenpom_dataframe(html_file_path=None, csv_file_path=None, save_data=False, year=None):
    """
    Creates a csv from an html file from kenpom and then loads the
    csv file as a dataframe to be used for analysis.

    Args:
        html_file_path(String): File path for the input .html file
        csv_file_path(String) File path for the output .csv file
        save_data(Boolean): If True, then the .csv file will be saved.
        year(int): The year to get stats from

    Raises:
        AssertionError: If `html_file_path` is not of type string.
        AssertionError: If `csv_file_path` is not of type string.
        AssertionError: If 'year' is not of type integer.

    Example:
        >>> load_kenpom_dataframe()
    """

    # Check that paths are both strings
    if html_file_path is not None:
        if not isinstance(html_file_path, six.string_types):
            raise AssertionError('Input file path must be a string.')
    if csv_file_path is not None:
        if not isinstance(csv_file_path, six.string_types):
            raise AssertionError('Output file path must be a string.')

    # Check that the year is an int
    if year is not None:
        if not isinstance(year, int):
            raise AssertionError('Year must be an integer.')

    # Load default values
    if html_file_path is None:
        html_file_path = '/Users/phil/Documents/Documents/College_Basketball/Data/kenpom.html'
    if csv_file_path is None:
        csv_file_path = '/Users/phil/Documents/Documents/College_Basketball/Data/kenpom.csv'

    # Check that files exits
    if os.path.exists(html_file_path):
        os.remove(html_file_path)
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)

    # Call functions to create pandas dataframe
    get_kenpom_html(html_file_path, year=year)
    kenpom_to_csv(html_file_path, csv_file_path)
    dataframe = load_csv(csv_file_path)

    # If not saving data deletes the created files
    os.remove(html_file_path)
    if not save_data:
        os.remove(csv_file_path)

    return dataframe


def get_kenpom_html(file_path=None, year=None):
    """
    Creates an html file from the kenpom webpage.

    Args:
        file_path(String): File path for the output .html file
        year(int): The year to get stats from

    Raises:
        AssertionError: If `file_path` is not of type string.
        AssertionError: If 'year' is not of type integer.

    Example:
        >>> get_kenpom_html()
    """

    # Check that the path is a strings
    if file_path is not None:
        if not isinstance(file_path, six.string_types):
            raise AssertionError('Output file path must be a string.')

    # Check that the year is an int
    if year is not None:
        if not isinstance(year, int):
            raise AssertionError('Year must be an integer.')

    # Load default values
    if file_path is None:
        file_path = '/Users/phil/Documents/Documents/College_Basketball/Data/kenpom.html'
    if year is None:
        year = datetime.datetime.now().year

    # Check that file exits
    if os.path.exists(file_path):
        os.remove(file_path)

    # Get the webpage html
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    http = urllib3.PoolManager()
    r = http.request('get', 'https://kenpom.com/index.php?y=' + str(year))

    with open(file_path, 'w') as fid:
        fid.write(r.data)


def kenpom_to_csv(input_file_path=None, output_file_path=None):
    """
    Creates a CSV file with data for each team from an html page from kenpom.

    Args:
        input_file_path(String): File path for the input .html file
        output_file_path(String) File path for the output .csv file

    Raises:
        AssertionError: If `input_file_path` is not of type string.
        AssertionError: If `output_file_path` is not of type string.
        AssertionError: If a file does not exist in the
            given `input_file_path`.

    Example:
        >>> kenpom_to_csv()
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
        input_file_path = '/Users/phil/Documents/Documents/College_Basketball/Data/kenpom.html'
    if output_file_path is None:
        output_file_path = '/Users/phil/Documents/Documents/College_Basketball/Data/kenpom.csv'

    # Check that fies exits
    if not os.path.exists(input_file_path):
        raise AssertionError('File does not exist at path %s' % input_file_path)
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # open kenpom html file and csv file
    kenpom_html = open(input_file_path, 'r')
    kenpom_csv = open(output_file_path, 'w')

    # Write the header
    kenpom_csv.write('Rank,Team,Conf,Wins,Losses,AdjEM,AdjO,AdjO Rank,AdjD,AdjD Rank,AdjT,AdjT Rank,Luck,Luck Rank,' +
                     'OppAdjEM,OppAdjEM Rank,OppO,OppO Rank,OppD,OppD Rank,NCSOS AdjEM,NCSOS AdjEM Rank\n')

    # Go through each line in the html file
    for line in kenpom_html:
        # Reset the list
        list = []

        # We only care about lines for each team
        if 'team=' in line:

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
                    item = line[start+1:i].replace('+', '')
                    if item != '' and item != ' ':
                        # If the item contains actual info save to list
                        list.append(line[start+1:i])
        
                i = i + 1

            # Check that the list is long enough
            if len(list) > 3:

                # Remove seeding from the data if necessary
                if list[2].isdigit():
                    del list[2]

                # Split W-L column into W column and L column
                list[3] = list[3].replace('-', ',')

                # Write the items to csv file
                kenpom_csv.write(','.join(list) + '\n')
