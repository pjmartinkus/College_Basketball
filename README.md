# College_Basketball

### Predicting the NCAA Tournament

The goal of this project is to use data driven methods to create a bracket for the NCAA Men's 
Basketball Tournament each year. Using data scraped from online resources, data cleaning techniques 
and machine learning methods, I hope to create the most accurate bracket possible before March 
Madness rolls around.

### Data

The data for this project currently comes from three sources. First, data for game scores for each
year are obtained from [sport reference's web site](https://www.sports-reference.com/cbb/boxscores/).
Additionally, team statistics are from Ken Pomeroy's college basketball statistics that are freely available at
[his website](https://kenpom.com), [Bart Torvik's website](http://www.barttorvik.com/) and from
[sport reference's web site](https://www.sports-reference.com/cbb/boxscores/). With these three data
sources I had a good variety of regular and tempo adjusted statistics to measure each team's overall performance.

### Overview of Methodologies

The high level plan of this project consists of five major steps: data extraction, data cleaning, 
feature generation, model selection, and making predictions.

As explained above, I used data from three sources: Kenpom,, T-Rank and Sports Reference. I created a python
function to obtain the data for each data source. These scripts make use of the great package 
[Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to help parse the html files
to extract the actual data. The data was then saved in [pandas dataframes](https://pandas.pydata.org)
for easy manipulation and handling.

The next step is to clean and combine the data sets together to create a single data source that can
be used in later steps. The most important data cleaning task was to match up the team name in each 
data set since each source uses different notation for school names. For example, one site could call 
University of North Carolina "UNC" and the other may call that same school "Carolina" and so on. The
other important task here was to join the datasets on both the school name and the year so that
each tuple in the resulting table would contain the score of a game in a given year as well as the yearly
statistics for each team that played in that game.

Once I had created a single table with records for each game containing the team statistics, I was able to create
some features. I decided to frame each tuple as a game between a favorite and an underdog, where I
included a feature for each team statistic in the Kenpom data: one for the favorite, one for the underdog
and lastly one for the difference between the two. Then each game could be labeled with a 0 when the favorite
won and a 1 when the underdog pulled off the upset. I thought this would be a useful way to look at the 
data because creating an NCAA Tournament bracket is all about looking for the right upsets.

At this point, I had a large data set containing feature vectors for each game of men's division one
college basketball going back to 2002. Before using all of this data, I wanted to refine it to only
include games that included tournament quality teams. I used covariate shift analysis to compare the training
feature vectors to the data set of actual tournament games and found an obvious difference between the two. By
looking at the most important feature in distinguishing between the data sets and trying some different
rules for filtering the complete data, I was able to reduce the training data to more closely resemble
actual NCAA Tournament games.

In addition to filtering the number of training feature vectors, I also wanted to reduce the number of features.
Since I had added the additional T-Rank and Sports Reference data sources this year, I knew there were many
redundant features covered in multiple source data sets. To this end, I looked at the correlations between
features and removed aas many aas I could without sacrificing too much potential information from the dropped
features.

Then, I trained several different machine learning models from the excellent
[scikit-learn](https://scikit-learn.org/stable/) python package. I started by tuning some parameters for
each model type using cross validation on a portion of the training data. Then, I compared the tuned models
using a variation of cross validation I like to call "leave one march out CV", meaning that for each fold,
I trained the model on all of the data except for the games from one year of March Madness, which was used as the
test set. Other than high values for standard accuracy metrics such as precision and recall, I found it 
was very important that the selected model would be able to give a probability score to help me make 
decisions about the most probable upsets. I also was conscious that my data was very noisy since it is
very difficult to predict how well teenagers playing basketball will actually perform, so I wanted to 
choose a more robust model. I settled on a logistic regression model to use as my predictor.

Finally, I was ready to create a bracket. I first trained the logistic regression model on all of the data
that I had collected. I then used it to make predictions on each game in the tournament. However, in order
to promote more upsets in my bracket, I added some rules to lower the threshold for predicting an upset
depending on some criteria like the round and seeding of the teams. Using this method, I created a bracket
and submitted it to ESPN's Tournament Challenge to test how well it performed.

### Last Year's Results
Unfortunately, my bracket did not do so well last year. My predicted winner, Illinois lost in the second
round to Loyola Chicago and Gonzaga was my only correct Final Four prediction. Overall, I correctly predicted
only 48% of the games in the tournament and 64% of the games that actually included my predicted winner
(in later rounds my predicted winner may have lost already), which was significantly down from the
previous season. Some of the error can be chalked up to the reduced non-conference schedule this past year
due to COVID, but the reality is that it is never easy to predict March Madness and that's whaat makes it
such a fun tournament to watch.

### Using the Code Available in the Repository
All of the code required to make my predictions are available in the python files and 
[Jupyter Notebooks](https://jupyter.org) included in the code and notebooks directories respectively. The
notebooks contain descriptions of what is going on and can hopefully provide enough guidance if you
intend to run the code yourself. Each of the main functions in the python code also contain descriptions
of what the function is intended to do, the input arguments and the final output. 
