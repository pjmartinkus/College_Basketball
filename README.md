# College_Basketball

### Predicting the NCAA Tournament

The goal of this project is to use data driven methods to create a bracket for the NCAA Men's 
Basketball Tournament each year. Using data scraped from online resources, data cleaning techniques 
and machine learning methods, I hope to create the most accurate bracket possible before March 
Madness rolls around.

### Data

The data for this project currently comes from two sources. First, team statistics for each year are 
retrieved from Ken Pomeroy's college basketball statistics that are freely available at 
[his website](https://kenpom.com). Additionally, data for game scores for each year are obtained 
from [sport reference's web site](https://www.sports-reference.com/cbb/boxscores/).

### Overview of Methodologies

The high level plan of this project consists of five major steps: data extraction, data cleaning, 
feature generation, model selection, and making predictions.

As explained above, I used data from two sources: Kenpom and Sport's Reference. I created a python
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
college basketball going back to 2002. Before using all of this data, I refined it to include
only games that included tournament quality teams. Then, I trained several different machine learning 
models from the excellent [scikit-learn](https://scikit-learn.org/stable/) python package. I used cross validation
to select the best model and eventually settled on an Adaboost model to use as my predictor because it had 
the best F1 score and the second best precision score. When predicting upsets, it is important to choose a classifier 
with high precision because I want to be a sure as possible that I am correct when the model predicts an upset. 
This is the case because any prediction made will affect the predictions in the later rounds. If the model makes too 
many upset predictions, it will most likely fail to predict the correct teams later in the bracket.

Finally, I was ready to create a bracket. I first trained the model on all of the data
that I had collected. I then used it to make predictions on each game in the tournament. However, in order
to promote more upsets in my bracket, I decided that I would call a game an upset if the model gave a 
probability score of 0.4985 or greater. Using this method, I created a bracket and submitted it to ESPN's 
Tournament Challenge to test how well it performed.

### Using the Code Available in the Repository
All of the code required to make my predictions are available in the python files and 
[Jupyter Notebooks](https://jupyter.org) included in the code and notebooks directories respectively. The
notebooks contain descriptions of what is going on and can hopefully provide enough guidance if you
intend to run the code yourself. Each of the main functions in the python code also contain descriptions
of what the function is intended to do, the input arguments and the final output. 
