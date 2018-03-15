import pandas


def update_kenpom(data):

    # Check that data is a dataframe
    if not isinstance(data, pandas.DataFrame):
        raise AssertionError('Input data must be a pandas dataframe.')

    # Lists of school names from scores and kenpom that don't match us perfectly
    scores = ['Texas-Arlington', 'Cleveland State', 'Penn State', "St. Peter's", 'Centenary (LA)', 'Alabama State',
     'Alabama-Birmingham', 'Morehead State', 'Jackson State', 'Southeast Missouri State', 'Texas A&M-Corpus Christi',
     'Weber State', 'Long Beach State', 'Citadel', 'UIC', 'Grambling', 'Arkansas-Little Rock', 'Youngstown State',
     'Jacksonville State', 'North Carolina-Asheville', 'Murray State', 'North Carolina State', 'Oklahoma State',
     'Cal State Northridge', 'Miami (OH)', 'UT-Martin', 'Albany (NY)', 'Gardner-Webb', 'Tennessee State',
     'Texas Christian', 'Texas-Rio Grande Valley', 'Alcorn State', 'Ball State', 'UC-Irvine', 'UCSB', 'Bethune-Cookman',
     'Idaho State', 'Bowling Green State', 'Florida International', 'Georgia State', 'Prairie View', 'Utah State',
     'Miami (FL)', 'Mississippi State', 'Morgan State', 'New Mexico State', 'San Jose State', 'Wichita State',
     'Wright State', 'Cal State Fullerton', 'Delaware State', 'Ole Miss', 'Savannah State', 'Kent State',
     'Portland State', 'UNC', "Saint Mary's (CA)", 'Middle Tennessee', 'Colorado State', 'UConn', 'St. Francis (NY)',
     'Iowa State', 'Loyola (IL)', 'Loyola (MD)', 'Arkansas-Pine Bluff', 'Kansas State', 'McNeese State',
     'North Carolina-Greensboro', 'Coppin State', 'Arkansas State', 'Troy', 'Virginia Military Institute',
     'Mississippi Valley State', 'Norfolk State', 'Oregon State', 'South Carolina State', "St. John's (NY)", 'UMass',
     'Louisiana', 'Pitt', 'Chicago State', 'Michigan State', 'LIU-Brooklyn', 'Indiana State', 'Texas State',
     'Arizona State', 'UC-Riverside', 'Fresno State', 'Missouri State', 'Nicholls State', 'North Carolina-Wilmington',
     'Appalachian State', 'Sacramento State', 'Montana State', 'Saint Francis (PA)', "St. Joseph's", 'Florida State',
     'Louisiana-Monroe', 'Ohio State', 'Maryland-Eastern Shore', 'Boise State', 'Birmingham-Southern', 'ETSU',
     'Illinois State', 'Northwestern State', 'University of California', 'Sam Houston State', 'San Diego State',
     'Washington State', 'Texas State', 'UC-Davis', 'Utah Valley', 'South Dakota State', 'Kennesaw State',
     'Missouri State', 'North Dakota State', 'Winston-Salem', 'South Carolina Upstate', 'Cal State Bakersfield',
     'Southern Illinois-Edwardsville', 'Nebraska-Omaha', 'UMass-Lowell', 'Texas-Rio Grande Valley',
     'IPFW', 'Arkansas-Little Rock']

    kenpom = ['UT Arlington', 'Cleveland St.', 'Penn St.', "Saint Peter's", 'Centenary', 'Alabama St.',
     'UAB', 'Morehead St.', 'Jackson St.', 'Southeast Missouri St.', 'Texas A&M Corpus Chris',
     'Weber St.', 'Long Beach St.', 'The Citadel', 'Illinois Chicago', 'Grambling St.', 'Arkansas Little Rock',
     'Youngstown St.', 'Jacksonville St.', 'UNC Asheville', 'Murray St.', 'North Carolina St.', 'Oklahoma St.',
     'Cal St. Northridge', 'Miami OH', 'Tennessee Martin', 'Albany', 'Gardner Webb', 'Tennessee St.',
     'TCU', 'Texas Pan American', 'Alcorn St.', 'Ball St.', 'UC Irvine', 'UC Santa Barbara', 'Bethune Cookman',
     'Idaho St.', 'Bowling Green', 'FIU', 'Georgia St.', 'Prairie View A&M', 'Utah St.',
     'Miami FL', 'Mississippi St.', 'Morgan St.', 'New Mexico St.', 'San Jose St.', 'Wichita St.',
     'Wright St.', 'Cal St. Fullerton', 'Delaware St.', 'Mississippi', 'Savannah St.', 'Kent St.',
     'Portland St.', 'North Carolina', "Saint Mary's", 'Middle Tennessee St.', 'Colorado St.', 'Connecticut',
     'St. Francis NY', 'Iowa St.', 'Loyola Chicago', 'Loyola MD', 'Arkansas Pine Bluff', 'Kansas St.',
     'McNeese St.', 'UNC Greensboro', 'Coppin St.', 'Arkansas St.', 'Troy St.', 'VMI',
     'Mississippi Valley St.', 'Norfolk St.', 'Oregon St.', 'South Carolina St.', "St. John's", 'Massachusetts',
     'Louisiana Lafayette', 'Pittsburgh', 'Chicago St.', 'Michigan St.', 'LIU Brooklyn', 'Indiana St.', 'Southwest Texas St.',
     'Arizona St.', 'UC Riverside', 'Fresno St.', 'Southwest Missouri St.', 'Nicholls St.', 'UNC Wilmington',
     'Appalachian St.', 'Sacramento St.', 'Montana St.', 'St. Francis PA', "Saint Joseph's", 'Florida St.',
     'Louisiana Monroe', 'Ohio St.', 'Maryland Eastern Shore', 'Boise St.', 'Birmingham Southern', 'East Tennessee St.',
     'Illinois St.', 'Northwestern St.', 'California', 'Sam Houston St.', 'San Diego St.', 'Washington St.',
     'Texas St.', 'UC Davis', 'Utah Valley St.', 'South Dakota St.', 'Kennesaw St.', 'Missouri St.',
     'North Dakota St.', 'Winston Salem St.', 'USC Upstate', 'Cal St. Bakersfield', 'SIU Edwardsville',
     'Nebraska Omaha', 'UMass Lowell', 'UT Rio Grande Valley', 'Fort Wayne', 'Little Rock']

    # Create the dictionary from the two lists
    names_dict = dict(zip(kenpom, scores))

    # Go through schools in the data and replace school names in the dictionary with the right name
    for i, row in data.iterrows():

        if row['Team'] in names_dict:
            data.iloc[i, data.columns.get_loc('Team')] = names_dict[row['Team']]

    return data





