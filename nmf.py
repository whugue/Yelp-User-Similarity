"""
(III) Use Non-Negative Matrix Factorization to Cluster Users
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from sklearn.utils.extmath import randomized_svd
from sklearn.manifold import TSNE


##Set Stopwords to Use in Vectorization.
sws = ENGLISH_STOP_WORDS.union([">NEW_REVIEW<","pittsburgh","pittsburgh pa","charlotte","charlotte nc","urbana-champaign","urbana","champaign",\
    "chicago","chicago il","urbana-champaign il","urbana il","champaign il","phoenix","phoenix az","las vegas","las","vegas","vegas nv","madison",\
    "madison wi","pa","nc","il","az","nv","love","like","awesome","bellagio","table","fork","spoon","knife","place","los","angeles","san","diego",\
    "san diego","loved","loved food"])



##Function to Subset Data by Sentiment Polarity (Either TextBlob or VADER Metrics)
def subset_on_polarity(df, metric, t):
    df["positive_sentiment"] = np.nan
    df.ix[df[metric]<0, "positive_sentiment"] = 0
    df.ix[df[metric]>=0, "positive_sentiment"] = 1

    return df.ix[abs(df[metric]) > t]


##Function to Run NMF - Vectorize using Binary Counts, Factorize, and Print Top 10 Topic Words and SVD Component Matrix
def nmf(text, n_components, n_iter=10):
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1,2), stop_words=sws, min_df=2, max_df=0.02,\
        token_pattern='\\b[a-z][a-z][a-z]+\\b', binary=True)                        #Fit Binary Count Vectorizer
    
    vectors = vectorizer.fit_transform(text)                                        #Fit Vocabulary & Transform Text into Binary Count Word Vector Space
    model = NMF(n_components=n_components, init="random", random_state=4444)        #Fit NMF Model
    topics = model.fit_transform(vectors)                                           #Use NMF To Convert Word Vector Space -> Topic Vector Space


    print "-----Top 10 Topic Words-----"
    print ""

    words = sorted([(i,v) for v,i in vectorizer.vocabulary_.items()])               #Print Top 10 Topic Words
    for r in model.components_:
        a = sorted([(v,i) for i,v in enumerate(r)], reverse=True)[0:10]
        print [words[e[1]] for e in a]
        print ""


    print "------Randomized SVD-------"
    print ""
    print randomized_svd(vectors, n_components=n_components, n_iter=n_iter, random_state=None)[1]

    return topics


##Function to Plot TNSE based on Topic Vector Space 
def plot_tsne():
    pass




##Run Functions
##Read In Sentence Level Data from Phoenix
print "Importing Data..."
sentences = pd.read_pickle("review_sentiments_all.pkl")


print "Subsetting Data on Sentiment Polarity..."
highly_polar = subset_on_polarity(sentences, "vd_polarity", 0.5)

print ""
print "Number of Highly Polar Users: "+str(highly_polar["user_id"].nunique())
print "Number of Highly Polar Reviews "+str(highly_polar["review_id"].nunique())
print "Number of Highly Polar Sentences: "+str(highly_polar.shape[0])
print ""

print highly_polar.positive_sentiment.value_counts(dropna=False)
print ""

print "Running NMF..."
print ""
#topic_vectors = nmf(df["sentence"], 20)







