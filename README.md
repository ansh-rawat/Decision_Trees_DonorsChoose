# Implementation of Decision Tree Classifier on DonorsChoose dataset.

Dataset link: https://www.kaggle.com/competitions/donorschoose-application-screening/data

Since the dataset has mostly the categorical features therefore, I've first encoded the all the features using one hot encoding and applied 
TFIDF Weighted Word2Vec on the feature 'essay' which is the most imporant criteria whether the funding of the project would be approved or not.

To apply Word2Vec I've used GloVe: https://en.wikipedia.org/wiki/GloVe

Tfidf w2v (w1,w2..) = (tfidf(w1) * w2v(w1) + tfidf(w2) * w2v(w2) + …) / (tfidf(w1) + tfidf(w2) + …)

Also, I've used SentimentIntensityAnalyzer from nltk library to get the sentiment scores for the feature essay and created 
additional features by adding them in my dataset. 

https://www.nltk.org/howto/sentiment.html

https://realpython.com/python-nltk-sentiment-analysis/#using-nltks-pre-trained-sentiment-analyzer


Hyperparameter tuning is done using GridSearchCV and found the best hyperparameters for the classifier using 3d plots.  

Used performance metrics like ROC/AUC, Confusion Matrix to check the performance of my model.

Finally used feature_importances_ attribute of Decision Tree which returns values of all the features based on their importance in getting preditions.
Then discarded all the features which have zero importance and trained the model using the ones which are non zero.

Wordcloud of the false positives for the feature essay, PDF's and box plot of some of the important features is plotted.


About Dataset:

DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. Right now, a large number of volunteers is needed to manually screen each submission before it's approved to be posted on the DonorsChoose.org website.

Next year, DonorsChoose.org expects to receive close to 500,000 project proposals. As a result, there are three main problems they need to solve:

How to scale current manual processes and resources to screen 500,000 projects so that they can be posted as quickly and as efficiently as possible.
How to increase the consistency of project vetting across different volunteers to improve the experience for teachers.
How to focus volunteer time on the applications that need the most assistance.
The goal of this kaggle competition is to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school. DonorsChoose.org can then use this information to identify projects most likely to need further review before approval.
