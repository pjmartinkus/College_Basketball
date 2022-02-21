---
jupyter:
  jupytext:
    formats: ipynb,markdown//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Post Tournament Evaluation and Metrics

In this notebook, we'll take a look at how our bracket did compared to the actual tournament results.

```python
# Import packages
import sys
sys.path.append('../')

import datetime
import pandas as pd
import collegebasketball as cbb
cbb.__version__
```

### Load in Scores from Last Season
Now that the season is complete, we can retrieve all of the scores for both evaluating our bracket this year and as more training data for next year.

```python
# Dates to search for games
year = 2021
start = datetime.date(year - 1, 11, 1)
end = datetime.date(year, 4, 10)

# Set up the path for this years scores
path = '../Data/Scores/'
path_regular = path + str(year) + '_season.csv'
```

```python
# cbb.load_scores_dataframe(start, end, csv_file_path=path_regular)
```

```python
data = pd.read_csv(path_regular)
data.head()
```

### Load in Predictions and Kenpom Data for This Season
In addition to the actual tournament game scores, we'll need our predictions and the pre-tournament Kenpom data to evaluate our bracket. The scores are obviously needed to verify when we were correct, but the Kenpom data is also necessary to determine which team was favored in each game since our model determines favorites using the Kenpom efficiency metric rather than tournament seeding.

Note that this year, one game was canceled due to COVID. While we could remove this game from the data, I'll just keep it in as an Oregon win since that's how many of the bracket sites, such as ESPN, scored that game.

```python
scores = cbb.filter_tournament(data)
predictions = pd.read_csv(f'../Data/predictions/predictions_{year}.csv')
kenpom = pd.read_csv(f'../Data/Kenpom/{year}_kenpom.csv')
kenpom = cbb.update_kenpom(kenpom)
kenpom.head(3)
```

```python
# Since Oregon VCU was cancelled due to COVID, we need to add fake score to indicate oregon won and moved on
scores.loc[scores['Away'] == 'Oregon', 'Home_Score'] = 0
scores.loc[scores['Away'] == 'Oregon', 'Away_Score'] = 1
scores[scores['Away'] == 'Oregon']
```

### Calculate Metrics

Now that we have all the necessary data, we can use a function to get all the metrics.

```python
# Run evaluation function on our data
cbb.post_tournament_eval(predictions, scores, kenpom)
```

Some notes on the various metrics above:
* **Games Contained Predicted Winner**: The number of games my bracket's predicted winner actually played in. In later rounds, my predicted winner may have already lost in a previous round so my predicted winner may not have even played in the game.
* **Total Upsets:** The number of actual upsets where an upset is defined as a team with a lower Kenpom efficiency score winning the game. Note this is referring the actual tournament results.
* **Upsets Predicted:** The number of games I predicted an upset based on the two teams I predicted to be playing in my bracket.
* **Games Containing Actual Upset Winner:** The number of games that were actual upsets where my bracket had the winning team predicted to be in the game.
* **Games Containing Predicted Upset Winner:** The number of games I predicted an upset where my predicted upset winner actually played. 
* **Correct Upsets:** The number of games where the actual winner was an underdog that I correctly predicted would win.
* **Correct Predicted Upsets:** The number of games where I predicted an upset and that team actually won the game.
* **Total Accuracy:** The fraction of all games where I correctly predicted the winner.
* **Upset Precision:** The fraction of all upsets I predicted that were correct.
* **Upset Recall:** The fraction of all actual upsets I predicted correctly.
* **Adj Accuracy:** The fraction of all games containing my predicted winner that I predicted correctly.
* **Adj Upset Precision:** The fraction of all games containing my predicted upset winner that I predicted correctly.
* **Adj Upset Recall:** The fraction of all actual upsets containing my predicted winner that I predicted correctly. 

The purpose of these "adjusted" stats are to adjust for the fact that in later rounds, it might not have even been possible to make a correct prediction based on previous errors. While I can see how you would want to know the full accuracy numbers for every game, I think these "adjusted" metrics provide a more accurate measure of performance on a game by game basis so that is why I've calculated them as well.


### How did the Bracket Perform this Year?

Unfortunately, not too well. While I can chalk some of the bad performance this year down to COVID, ultimately predicting the NCAA Tournament will never be easy and bad years are bound to happen. I think the game cancellations from COVID did have a pretty big impact on the reliability of our team metrics this year because many of them occurred during non-conference play. This lack of data between conferences made it even harder than usual to adjust efficiency metrics like Kenpom based on strength of schedule because there weren't many data points to compare teams in different conferences. For example, the Big Ten had strong Kenpom stats across the board. Perhaps the conference's bad performance was a results of inflated ratings from conference play between teams that were never actually tested against other conferences. 

Overall, I'll need to go back to the drawing board and make some adjustments in the future, particularly to how I pick the actual tournament winners. Since going exactly with whether or not the model predicts an upset results in far too few upset predictions, I lowered the threshold of the predicted probability required to pick an upset. I have yet to come up with a satisfying method for dealing with this issue and surely it is part of the reason for the poor performance this year.

```python

```
