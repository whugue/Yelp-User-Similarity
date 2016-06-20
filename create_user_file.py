
"""
Classify Yelp Review Sentences and Aggregate up to the User Level
"""


import numpy as np
import pandas as pd
import pickle

from sklearn.manifold import TSNE
np.set_printoptions(suppress=True)


#Function to Read in Pickled Classifiers 
def open_pickle(path):
    with open(path) as f:
        out = pickle.load(f)
    return out


##Function to Read in Each Location's Yelp Data and Add to Corpus
def read_in_yelp(base, inpath):
    df = pd.read_pickle(inpath)
    return pd.concat([base, df], axis=0)


##Function to Classify Yelp Review Sentences by Topic
def classify_sentences(vectorizer, df, clf, topic):
    X = vectorizer.transform(df["sentence"]) #Transform Yelp Data onto Word Vector Space
    p = pd.Series(clf.predict(X), name=topic)

    return pd.concat([df, p], axis=1)


##Function Aggregate Topic Tagged Sentences to User Level -> Sum up number of sentences by topic, relevant, and total
def create_user_file(df):
    return df.groupby(by="user_id", as_index=False)["topic_food","topic_service","topic_ambience","topic_value","relevant","total"].sum()

    #Since we aren't looking at distribution across space to figure out similar users, no need to do this:
    #user = df.groupby(by=["location","user_id"], as_index=False)[topics].sum() #Sum Topic Flags up to Location-User Level
    #user.sort_values(by="relevant", inplace=True) 
    #user.drop_duplicates(subset="user_id", keep="last", inplace=True)
    #return user

##Function to Pickle Objects for Further Analysis:
def save_pickle(item, outpath):
    with open(outpath, "wb") as f:
        pickle.dump(item, f)


##Run Functions
print "Reading In Vectorizer and Classifiers..."
binary_vectorizer = open_pickle("binary_vectorizer.pkl")
lsvm_food = open_pickle("lsvm_food.pkl")
lsvm_service = open_pickle("lsvm_service.pkl")
lsvm_ambience = open_pickle("lsvm_ambience.pkl")
lsvm_value = open_pickle("lsvm_value.pkl")


##Initalize Container for Yelp Data (Pandas DF)
print "Reading in Yelp Review Data..."
yelp = pd.DataFrame()
yelp = read_in_yelp(yelp, "review_sentences_charlotte.pkl")
yelp = read_in_yelp(yelp, "review_sentences_pittsburgh.pkl")
yelp = read_in_yelp(yelp, "review_sentences_madison.pkl")
yelp.reset_index(inplace=True, drop=True)                                   #Reset Index after Stacking

print "Total Number of Yelp Users:     ", yelp["user_id"].nunique()         #Print Number of Users in Yelp Masterfile
print "Total Number of Yelp Reviews:   ", yelp["review_id"].nunique()       #Print Number of Reviews in Yelp Masterfile
print "Total Number of Yelp Sentences: ", yelp.shape[0]                     #Print Number of Sentences in Yelp Masterfile


##Classify Sentences
print "Classifying Yelp Review Data..."
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_food, "topic_food")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_service, "topic_service")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_ambience, "topic_ambience")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_value, "topic_value")
yelp["relevant"] = yelp[["topic_food", "topic_service", "topic_ambience", "topic_value"]].max(axis=1)   #Create Flag for Topic-Relevant Sentences (total)
yelp["total"] = 1


##Sum Up to User Level
print "Aggregating to User Level..."
user = create_user_file(yelp)

print "Size of User-Level File: ", user.shape[0]


#Pickle User Level File
print "Pickling User-Level File..."
save_pickle(user, "yelp_review_user.pkl")













