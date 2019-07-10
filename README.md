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

As this project progresses, I plan to include team data from more sources. I have already wrote the 
code required to scrape team statistics data from 
[Sport's Reference](https://www.sports-reference.com/cbb/seasons/2019-school-stats.html) and [Bart 
Torvik's website](http://www.barttorvik.com/) and will include them in next year's predictions. 
In the future, I hope to include more data sources including other advanced season metrics, player 
specific statistics, recruiting rankings and more.

### Overview of Methodologies

The high level plan of this project consists of five major steps: data extraction, data cleaning, 
feature generation, model selection, and making predictions. Below I give a brief explanation of
each step, but for more information, check out the project summaries for 
[2018](https://github.com/pjmartinkus/College_Basketball/tree/master/Version%20History/2018/Results) and
[2019](https://github.com/pjmartinkus/College_Basketball/tree/master/Version%20History/2019/Results).

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
models from the excellent [scikit-learn](https://scikit-learn.org/stable/) python package. I used a variation
of cross validation I like to call "leave one march out CV", meaning that for each fold, I trained the
model on all of the data except for the games from one year of March Madness, which was used as the
test set. Other than high values for standard accuracy metrics such as precision and recall, I found it 
was very important that the selected model would be able to give a probability score to help me make 
decisions about the most probable upsets. I also was conscious that my data was very noisy since it is
very difficult to predict how well teenagers playing basketball will actually perform, so I wanted to 
choose a more robust model. I settled on a logistic regression model to use as my predictor.

Finally, I was ready to create a bracket. I first trained the logistic regression model on all of the data
that I had collected. I then used it to make predictions on each game in the tournament. However, in order
to promote more upsets in my bracket, I decided that I would call a game an upset if the model gave a 
probability score of 0.4 or greater. Using this method, I created a bracket and submitted it to ESPN's 
Tournament Challenge to test how well it performed.

### Last Year's Results
Last year, in the 2019 NCAA Tournament, I successfully 
[predicted](https://github.com/pjmartinkus/College_Basketball/blob/docs_summary/Version%20History/2019/Results/Tournament%20Challenge%20-%20ESPN%20-%202019.pdf) 
that Virginia would win the tournament, though I should note they were a top team per the Kenpom data
I used in this project despite their historic loss in 2018. Overall, I correctly predicted 77% of the
games in the tournament and 83% of the games that actually included my predicted winner (in later rounds 
my predicted winner may have lost already). 

As a fun test of how my bracket faired against the rest of the country, I submitted it to ESPN's
Tournament Challenge and it scored 1370 points, was ranked about 121 thousandth, which left it in 
the 99th percentile. These results are of course skewed by the way ESPN counts points since I had
the correct champion, but I think it was a fun experiment none the less.

### Using the Code Available in the Repository
All of the code required to make my predictions are available in the python files and 
[Jupyter Notebooks](https://jupyter.org) included in the code and notebooks directories respectively. The
notebooks contain descriptions of what is going on and can hopefully provide enough guidance if you
intend to run the code yourself. Each of the main functions in the python code also contain descriptions
of what the function is intended to do, the input arguments and the final output. 
