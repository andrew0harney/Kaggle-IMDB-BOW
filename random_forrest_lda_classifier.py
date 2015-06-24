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
from gensim import corpora, models
import pprint
from sklearn.externals import joblib
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ALLOWED_WORDS = set(nltk_words.words()) - set(stopwords.words())
match_non_word_chars = re.compile(r'[%s0-9]' % string.punctuation)
match_whitespace = re.compile(r'\s+')

def review_to_words(review):
    soup = BeautifulSoup(review)
    [x.extract() for x in soup(['script','style'])]
    text = soup.getText()
    return filter(lambda word: word in ALLOWED_WORDS , match_whitespace.sub(' ', match_non_word_chars.sub('', text.lower())).split(' '))


def convert_lda_output(prediction):
    output = [0,0]
    for i,x in prediction:
        output[i] = x
    return output


pprint.pprint('Loading train/test set')
train = pd.read_csv(os.path.join(os.getcwd(), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
train = train.reindex(np.random.permutation(len(train))).reset_index()
test = train[20000:]
train = train[:20000]


vectoriser = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)


#Train LDA for positive and negative topics
pprint.pprint('Cleaning training data')
clean_train_reviews = [review for review in train['review'].values.tolist()]
reviews = [" ".join(review_to_words(review)) for review in clean_train_reviews]
train_data_features = vectoriser.fit_transform(reviews).toarray()

vocab = corpora.Dictionary([vectoriser.get_feature_names()])
tokenised_train = [vocab.doc2bow(review.split()) for review in reviews]

pprint.pprint('Training LDA')
lda = models.ldamodel.LdaModel(corpus=tokenised_train, id2word=vocab, num_topics=2, passes=100)
lda.save(fname=os.path.join(os.getcwd(), 'models', 'lda.model'))


pprint.pprint('Adding LDA features to training')
train_lda_out = lda[tokenised_train]
train_topics = np.array([convert_lda_output(prediction) for prediction in train_lda_out])
train_data_features = np.hstack((train_data_features, train_topics))

#Train sentiment
pprint.pprint('Training sentiment')
forest = RandomForestClassifier(n_estimators = 150)
forest = forest.fit(train_data_features, train["sentiment"])
joblib.dump(forest, os.path.join(os.getcwd(), 'models', 'forest.model'))

#Test
pprint.pprint('Cleaning test data')
clean_test_reviews = [" ".join(review_to_words(review)) for review in test['review'].values.tolist()]
tokenised_test = [vocab.doc2bow(review.split()) for review in clean_test_reviews]

pprint.pprint('Adding LDA features to testing')
test_lda_out = lda[tokenised_test]
test_topics = np.array([convert_lda_output(prediction) for prediction in test_lda_out])
test_data_features = np.hstack((vectoriser.transform(clean_test_reviews).toarray(), test_topics))

pprint.pprint('Predicting')
result = forest.predict(test_data_features)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result, "actual":test["sentiment"]} )
output.to_csv(os.path.join(os.getcwd(), 'results', 'bowRandomForestResults.csv'), index=False, quoting=3)

pprint.pprint('ROC_AUC : %f' % roc_auc_score(test['sentiment'], result))
