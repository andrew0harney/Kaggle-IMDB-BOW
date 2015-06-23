#!/usr/bin/env python
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from nltk.corpus import words as nltk_words
from nltk.corpus import stopwords
import pandas as pd
import re
from bs4 import BeautifulSoup
import string
import numpy as np

ALLOWED_WORDS = set(nltk_words.words()) - set(stopwords.words())
match_non_word_chars = re.compile(r'[%s0-9]' % string.punctuation)
match_whitespace = re.compile(r'\s+')

def review_to_words(review):
    soup = BeautifulSoup(review)
    [x.extract() for x in soup(['script','style'])]
    text = soup.getText()
    return filter(lambda word: word in ALLOWED_WORDS , match_whitespace.sub(' ', match_non_word_chars.sub('', text.lower())).split(' '))

train = pd.read_csv(os.path.join(os.getcwd(), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
train = train.reindex(np.random.permutation(len(train))).reset_index()
test = train[20000:]
train = train[:20000]


vectoriser = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 1000)

#Train random forest with bag of words and sentiment
clean_train_reviews = [" ".join(review_to_words(review)) for review in train['review'].values.tolist()]

train_data_features = vectoriser.fit_transform(clean_train_reviews).toarray()
count = np.sum(train_data_features, axis=0)

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features, train["sentiment"])

#Test
clean_test_reviews = [" ".join(review_to_words(review)) for review in test['review'].values.tolist()]
test_data_features = vectoriser.transform(clean_test_reviews).toarray()
result = forest.predict(test_data_features)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result, "actual":test["sentiment"]} )
output.to_csv(os.path.join(os.getcwd(), 'results', 'bowRandomForestResults.csv'), index=False, quoting=3)

print roc_auc_score(test['sentiment'], result)