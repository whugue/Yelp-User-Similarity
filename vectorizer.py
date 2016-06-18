"""
(III) Fit Vectorizers on Entire Corpus of ABSA & Yelp Review Data
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


##Functions to Read in Series of Sentences from ABSA or Yelp Data and Add to Full Text Corpus
def add_to_corpus(corpus, inpath):
	sentences = pd.read_pickle(inpath)["sentence"]
	return pd.concat([corpus, sentences], axis=0)


##Functions to Fit Binary Count, Multinomial Count, and TF-IDF Vectorizers on Entire Corpus Vocabulary
def fit_binary_vectorizer(corpus):
	vectorizer = CountVectorizer(ngram_range=(1,1), min_df=2, max_df=0.2, stop_words="english", token_pattern="\\b[a-z][a-z][a-z]+\\b", binary=True)
	vectorizer.fit(corpus)

	return vectorizer
	
def fit_count_vectorizer(corpus):
	vectorizer = CountVectorizer(ngram_range=(1,1), min_df=2, max_df=0.2, stop_words="english", token_pattern="\\b[a-z][a-z][a-z]+\\b")
	vectorizer.fit(corpus)

	return vectorizer

def fit_tfidf_vectorizer(corpus):
	vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=2, max_df=0.2, stop_words="english", token_pattern="\\b[a-z][a-z][a-z]+\\b")
	vectorizer.fit(corpus)

	return vectorizer


##Function to Pickle Vectorizers
def pickle_vectorizer(vectorizer, outpickle):
	with open(outpickle, "wb") as f:
		pickle.dump(vectorizer, f)


##Run Functions
##Create Full ABSA+Yelp Corpus
print "Creating Full Corpus..."
review_corpus = pd.Series() #Initialize Corpus as Empty Pandas Series

review_corpus = add_to_corpus(review_corpus, "all_absa_data.pkl")
review_corpus = add_to_corpus(review_corpus, "review_sentences_charlotte.pkl")
review_corpus = add_to_corpus(review_corpus, "review_sentences_madison.pkl")
review_corpus = add_to_corpus(review_corpus, "review_sentences_pittsburgh.pkl")

review_corpus.drop_duplicates(inplace=True) #Remove Duplicate Review Sentences from Corpus

print "Created Full Corpus of "+str(review_corpus.shape[0])+" sentences."
print ""


##Fit Vectorizers to Corpus
print "Creating Binary Vectorizer..."
binary_vectorizer = fit_binary_vectorizer(review_corpus)

print "Creating Count Vectorizer..."
count_vectorizer = fit_count_vectorizer(review_corpus)

print "Creating TF-IDF Vectorizer..."
tfidf_vectorizer = fit_tfidf_vectorizer(review_corpus)
print ""


#Pickle Vectorizers
print "Pickling Binary Vectorizer..."
pickle_vectorizer(binary_vectorizer, "binary_vectorizer.pkl")

print "Pickling Count Vectorizer..."
pickle_vectorizer(count_vectorizer, "count_vectorizer.pkl")

print "Pickling TF-IDF Vectorizer..."
pickle_vectorizer(tfidf_vectorizer, "tfidf_vectorizer.pkl")



