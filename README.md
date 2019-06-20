# Housing Prices Competition for Kaggle Learn Users

Overview of the Kaggle competition can be found via the following link:
*https://www.kaggle.com/c/home-data-for-ml-course/overview*


The goal of the competition was to forecast the housing prices. The competition was a good way for me to begin my journey learning python and dip my toes into boosting algorithims. 

I fitted an XGBoost model to the data and ran 10-fold cross validation on it to select the best parameter for n_estimators, the number of trees to fit, in the model. The value chosen as a result was 750. A similar approach was used to select the learning rate of the algorithm.

With this simplistic approach, I was able to reach the leaderboard as high as 313th place out of 6,845 people as of July 2019, placing me in the 95.4 percentile or top 4.5%.