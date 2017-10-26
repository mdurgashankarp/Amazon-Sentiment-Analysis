from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import lda
import numpy as np 
import pandas as pd 
import csv

# Creation of iterable function
def articles_in_csv():
    a = open('/Users/shakunthala/Desktop/reviews.csv', 'rt', encoding="ISO-8859-1")
    for row in csv.reader(a, delimiter=',', quotechar='"' , dialect='excel'):
        yield row[9]

n_features = 1000
n_topics = 10
n_top_words = 25

t0 = time()
print("Loading dataset and extracting TF-IDF features...")

# We don't consider the English stop words and words that cover more than 95% of the texts
vectorizer = TfidfVectorizer(max_df=0.95, 
                             min_df=2, 
                             max_features=n_features,
                             stop_words='english')

tfidf = vectorizer.fit_transform(articles_in_csv())
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model

nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
'''lda_model = lda.LDA(n_topics, n_iter=1000, random_state=1)
lda_model.fit(tfidf)
topic_word = lda_model.topic_word_
n_top_words = 25
for i, topic_dist in enumerate(topic_word):
	topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {}: {}'.format(i, ' '.join(topic_words)))'''
print("done in %0.3fs." % (time() - t0))

feature_names = vectorizer.get_feature_names()

# Print of the results
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
