
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
    topics = ["topic_food", "topic_service", "topic_ambience", "topic_value", "relevant", "total"]
    
    user = df.groupby(by=["location","user_id"], as_index=False)[topics].sum() #Sum Topic Flags up to Location-User Level
    
    user.sort_values(by="relevant", inplace=True) 
    user.drop_duplicates(keep="last", inplace=True) #De-Duplicate-If one user reviewed in multiple cities, chose large corpus
    
    return user


#Subset Out Noise (e.g. Very Low or High Relevant Sentences)
def subset_users(df, var, min, max):
    return df[(df[var] >= min) & (df[var] <= max)]


#Calculate tSNE values for Each User to Reduce Dimentions to 2 (for plotting/ clustering)
def tSNE(df, variables):
    model = TSNE(n_components=2, random_state=4444)

    reduced = model.fit_transform(df[variables])
    reduced = pd.DataFrame(reduced, columns=["tSNE_1","tSNE_2"])

    return pd.concat([df, reduced], axis=1)


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

yelp.reset_index(inplace=True, drop=True)                   #Reset Index after Stacking
print "Yelp Sentences Read In: ", yelp.shape[0]             #Print Number of Sentences (~1.7M)


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

print "Number of Users Classified: ", user.shape[0]


#Pickle User Level File
print "Pickling User-Level File..."
save_pickle(user, "yelp_review_user.pkl")













