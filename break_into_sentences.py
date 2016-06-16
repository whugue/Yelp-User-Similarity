
"""
(I) Break Yelp Reviews into Component Sentences
    (1) Subset to Only Restaurant Reviews in Charlotte, NC
    (2) Split all reviews into sentences. Output as One Pandas Dataframe
"""

import json
from collections import defaultdict

import pandas as pd
from nltk.tokenize import sent_tokenize

from pymongo import MongoClient
from bson import Code


#Boot Up Mongo DB and Read in Collections
client =            MongoClient()
business =          client.yelp.business
review =            client.yelp.review


##Function to get List of all Business IDs for Restaurants in Six U.S. Metropolitan Areas to Analyze
def get_business_ids(location, mongo_query):
    print "Returning Business IDs for "+location+"..."
    business_ids = []

    cursor = business.find(mongo_query)

    for each in cursor:
        business_ids.append(each["business_id"])

    return business_ids


##Function to Break Each Review into it's component sentences & calculate TextBlob & VADER sentiment polarity of those sentences
def create_sentence_df(in_collection, location, mongo_query):
    print "Creating Cursor for "+location+"..."
    df = []
    business_ids=get_business_ids(location, mongo_query)
    cursor = in_collection.find({"business_id": {"$in": business_ids}})

    #Split Each Review into it's component sentences with one row = one review sentence
    print "Spliting Reviews into Sentences for "+location+"..."
    for review in cursor:
        sentences = sent_tokenize(review["text"])

        for sentence in sentences:
            row=defaultdict(str)

            row["user_id"] = review["user_id"]  #User ID
            row["review_id"] = review["_id"]    #Review ID (Object ID from MongoDB)
            row["location"] = location          #Location (from function input)
            row["sentence"] = sentence

            df.append(row)
    print "DF Created for: "+location
    return pd.DataFrame(df)



##Run Functions!
restaurants = json.loads(open("data/yelp/restaurants.json","r+").read())["food_places"]


##Create Dataframes
madison = create_sentence_df(review, "Madison, WI", {"state":"WI", "categories": {"$in": restaurants}})                         #Madison, WI
pittsburg = create_sentence_df(review, "Pittsburg, PA", {"state": "PA", "categories": {"$in": restaurants}})                    #Pittsburgh, PA
charlotte = create_sentence_df(review, "Charlotte, NC", {"state": {"$in": ["NC","SC"]}, "categories": {"$in": restaurants}})    #Charlotte, NC
urbana = create_sentence_df(review, "Urbana-Champaign, IL", {"state": "IL", "categories": {"$in": restaurants}})                #Urbana-Champaign, IL
phoenix = create_sentence_df(review, "Phoenix, AZ", {"state": "AZ", "categories": {"$in": restaurants}})                        #Phoenix, AZ
las_vegas = create_sentence_df(review, "Las Vegas, NV", {"state": "NV", "categories": {"$in": restaurants}})                    #Las Vegas, NV


#Pickle Dataframes for Upload onto AWS for Further Analysis
madison.to_pickle("data/yelp/dataframes/review_sentences_madison.pkl")
pittsburg.to_pickle("data/yelp/dataframes/review_sentences_pittsburgh.pkl")
charlotte.to_pickle("data/yelp/dataframes/review_sentences_charlotte.pkl")
urbana.to_pickle("data/yelp/dataframes/review_sentences_urbana.pkl")
phoenix.to_pickle("data/yelp/dataframes/review_sentences_phoenix.pkl")
las_vegas.to_pickle("data/yelp/dataframes/review_sentences_las_vegas.pkl")









