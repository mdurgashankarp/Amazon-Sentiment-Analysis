get_ipython().magic('matplotlib inline')

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import os
from IPython.core.display import display, HTML
    
if not os.path.isfile('C:/Users/mpdur/Downloads/amazon-fine-food-reviews/database.sqlite'):
    display(HTML("""
        <h3 style='color: red'>Dataset database missing!</h3><h3> Please download it
        <a href='https://www.kaggle.com/snap/amazon-fine-food-reviews'>from here on Kaggle</a>
        and extract it to the current directory.con = sqlite3.connect('../input/database.sqlite')

pd.read_sql_query("SELECT * FROM Reviews LIMIT 3", con)
          """))
    raise(Exception("missing dataset"))

con = sqlite3.connect('C:/Users/mpdur/Downloads/amazon-fine-food-reviews/database.sqlite')
pd.read_sql_query("SELECT * FROM Reviews", con)

import matplotlib.pyplot as plt
import seaborn as sns

messages = pd.read_sql_query("""
SELECT 
  Score, 
  Summary, 
  HelpfulnessNumerator as VotesHelpful, 
  HelpfulnessDenominator as VotesTotal, Text
FROM Reviews 
WHERE Score != 3 and VotesTotal > 10""", con)

messages.describe()
messages.info()
sns.pairplot(messages)
sns.heatmap(messages.corr(),linecolor='green')

import string
from nltk.corpus import stopwords

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['Text'])
print(len(bow_transformer.vocabulary_))
messages_bow = bow_transformer.transform(messages['Text'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['Score'])

print('predicted:', spam_detect_model.predict(messages_tfidf)[0])
print('expected:', messages.Score[3])

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


from sklearn.metrics import classification_report
print (classification_report(messages['Score'], all_predictions))

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['Text'], messages['Score'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))

