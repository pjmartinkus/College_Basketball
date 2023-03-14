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
year = 2022
start = datetime.date(year - 1, 11, 1)
end = datetime.date(year, 4, 10)

# Set up the path for this years scores
path = '../Data/Scores/'
path_regular = path + str(year) + '_season.csv'
```

```python
cbb.load_scores_dataframe(start, end, csv_file_path=path_regular)
```

```python
data = pd.read_csv(path_regular)
data.head()
```

### Load in Predictions and Kenpom Data for This Season
In addition to the actual tournament game scores, we'll need our predictions and the pre-tournament Kenpom data to evaluate our bracket. The scores are obviously needed to verify when we were correct, but the Kenpom data is also necessary to determine which team was favored in each game since our model determines favorites using the Kenpom efficiency metric rather than tournament seeding.

```python
scores = cbb.filter_tournament(data)
predictions = pd.read_csv(f'../Data/predictions/predictions_{year}.csv')
kenpom = pd.read_csv(f'../Data/Kenpom/{year}_kenpom.csv')
kenpom = cbb.update_kenpom(kenpom)
kenpom.head(3)
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

To start off with, I'm excited to say that the bracket successfully predicted the winner for the third time out of four years. It's honestly incredible how frequently this bracket has predicted the winner even if you account for some favorites winning it over the last few years and I honestly don't expect this trend to continue.

Overall, it was a pretty average year as far as the metrics go. The overall accuracy and adjusted accuracy were down compared to previous seasons, but the upset precision and recall metrics were overall slightly up. This is mostly due to there being more upsets than previous seasons, so I think it's fair to say that the model actually was better than previous years, but the frequent number of upsets caused overall accuracy to be lower in spite of that.

Unfortunately, I haven't found much time to make many improvements for next year, but I'm looking forward to seeing how the bracket performs next tournament.

```python

```
