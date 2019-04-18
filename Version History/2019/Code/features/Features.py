

def teams(row, columns):
    # Get the team names
    home = row[columns.get_loc('Team_Home')]
    away = row[columns.get_loc('Team_Away')]

    # Find out which team is favored by kenpom
    home_favored = False
    if row[columns.get_loc('AdjEM_Home')] > row[columns.get_loc('AdjEM_Away')]:
        home_favored = True

    # Return favored team first, then underdog, then difference
    if home_favored:
        return home, away
    else:
        return away, home


def win_loss(row, columns):
    # Calculate winning percentages
    home = row[columns.get_loc('Wins_Home')] / \
           float(row[columns.get_loc('Losses_Home')] + row[columns.get_loc('Wins_Home')])
    away = row[columns.get_loc('Wins_Away')] / \
           float(row[columns.get_loc('Losses_Away')] + row[columns.get_loc('Wins_Away')])

    # Depending on who is favored, return the results
    return favored_team(home, away, row, columns)


def feature_difference(row, columns, feature):
    # Calculate winning percentages
    home = row[columns.get_loc(feature + '_Home')]
    away = row[columns.get_loc(feature + '_Away')]

    # Depending on who is favored, return the results
    return favored_team(home, away, row, columns)


def favored_team(home, away, row, columns):
    # Find out which team is favored by kenpom
    home_favored = False
    if row[columns.get_loc('AdjEM_Home')] > row[columns.get_loc('AdjEM_Away')]:
        home_favored = True

    # Return favored team first, then underdog, then difference
    if home_favored:
        return home, away, home - away
    else:
        return away, home, away - home
