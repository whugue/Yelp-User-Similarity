"""
(II) Calculate Sentiment Polarity for Each Review Sentence
"""

import pandas as pd

from textblob import TextBlob
from vaderSentiment import vaderSentiment as vader

##Read In Emptpy Dataframe Container
sentences = pd.DataFrame()


##Stack Datasets from Various Cities into a Single Dataframe
def stack_df(location):
	print "Importing Data: "+location

	infile = "review_sentences_"+location+".pkl"
	df = pd.read_pickle(infile)

	return pd.concat([sentences, df], axis=0)


##Define Functions to Calculate TextBlob and VADER Sentiment Polarity
def text_blob_sentiment(text):
    return TextBlob(text).sentiment.polarity

def vader_sentiment(text):
    text = text.encode("ascii", errors="ignore")
    return vader.sentiment(text)["compound"]

def calculate_polarity(df):
	df["tb_polarity"] = df["sentence"].apply(text_blob_sentiment)
	df["vd_polarity"] = df["sentence"].apply(vader_sentiment)

	return df


##Run Functions
sentences = stack_df("charlotte")
sentences = stack_df("pittsburgh")
sentences = stack_df("madison")

print ""
print "Number of Users: "+str(sentences["user_id"].nunique())
print "Number of Reviews:"+str(sentences["review_id"].nunique())
print "Number of Sentences: "+str(sentences.shape[0])
print ""

print "Calculating Sentiment..."
print ""

sentiments = calculate_polarity(sentences)
sentiments.to_pickle("review_sentiments_all.pkl")

print sentiments.head(10)