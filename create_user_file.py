
"""
Classify Yelp Review Sentences and Aggregate up to the User Level
"""

##TO DO: tally both meaningful and non-meaningful sentences.


import numpy as np
import pandas as pd
import pickle


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


##Function to Create User Level File: Calculate N/% Sentences per Topic
def create_user_file(df):
    topics = ["topic_food","topic_service","topic_ambience","topic_value","total"]
    
    user = df.groupby(by=["location","user_id"], as_index=False)[topics].sum() #Sum Topic Flags up to Location-User Level
    
    user.sort_values(by="total", inplace=True) 
    user.drop_duplicates(keep="last", inplace=True) #De-Duplicate-If one user reviewed in multiple cities, chose large corpus
    
    user["pct_food"] = user["topic_food"] / user["total"] #Calculate Percentages
    user["pct_service"] = user["topic_service"] / user["total"]
    user["pct_ambience"] = user["topic_ambience"] / user["total"]
    user["pct_value"] = user["topic_value"] / user["total"]
    
    return user


##Function to Pickle Created Objects
def save_pickle(item, path):
    with open(path, "wb") as f:
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

yelp.reset_index(inplace=True, drop=True)   #Reset Index after Stacking


##Classify Sentences
print "Classifying Yelp Review Data..."
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_food, "topic_food")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_service, "topic_service")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_ambience, "topic_ambience")
yelp = classify_sentences(binary_vectorizer, yelp, lsvm_value, "topic_value")
yelp[]

##Sum Up to User Level
print "Aggregating to User Level..."
user = create_user_file(yelp)


#Pickle
print "Pickling User-Level File..."
save_pickle(user, "yelp_review_user.pkl")



