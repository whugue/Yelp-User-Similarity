
"""
Script:     01-Preprocess-Yelp-Data.py
Purpose:    (a) Filter Yelp Academic Dataset to Only Restaurants Reviews in Charlotte NC, Madison WI, and Pittsburgh PA
            (b) Break review data into component sentences so 1 record = 1 sentence
Input:      Business and Review MongoDB Collections 
            (see data/README.md for more info on how to load the raw Yelp data into Mongo)
Output:     data/yelp/dataframes/review_sentences_madison.pkl
            data/yelp/dataframes/review_sentences_pittsburgh.pkl
            data/yelp/dataframes/review_sentences_charlotte.pkl (Pandas Dataframes)
"""

import json
from collections import defaultdict

import pandas as pd
from nltk.tokenize import sent_tokenize

from pymongo import MongoClient


#Boot Up Mongo DB and Read in Collections
client =            MongoClient()
business =          client.yelp.business
review =            client.yelp.review


"""
Function filter Yelp businesses to only RESTUARANTS in a given city (e.g. Charlotte, NC)
location:       Location of Yelp Businessed pulled by input mongo_query (e.g. Charlotte, NC)
mongo_query:    Mongo query to filter all Yelp businesses to ONLY restauarants in a given area
RETURNS:        List of Business IDs for restaurants in a given city (as specified by input mongo_query)
"""

def get_business_ids(location, mongo_query):
    print "Returning Business IDs for "+location+"..."
    business_ids = []

    cursor = business.find(mongo_query)

    for each in cursor:
        business_ids.append(each["business_id"])

    return business_ids


"""
Function to break each review into it's compotnent sentences

in_collection:  MongoDB collection containing Yelp review data
location:       Location of Yelp Businessed pulled by input mongo_query (e.g. Charlotte, NC) (to impliment above function)
mongo_query:    List of Business IDs for restaurants in a given city (as specified by input mongo_query) (to impliment above function)

RETURNS:        Pandas DF of restaurant reviews for given location, where 1 row = 1 sentence
"""

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




##Run Above Function to Filter & Clean Yelp Review Data for Analysis

##This JSON file contains a list of all Yelp business categories considered as "restaurants for this analysis."
##These categories were derived by hand from a list of all possible business categories in the raw Yelp data
restaurants = json.loads(open("data/yelp/restaurants.json","r+").read())["food_places"]


##Filter to Resturant Reviews for [city] and split out into component sentences
madison = create_sentence_df(review, "Madison, WI", {"state":"WI", "categories": {"$in": restaurants}})                        
pittsburg = create_sentence_df(review, "Pittsburg, PA", {"state": "PA", "categories": {"$in": restaurants}})                 
charlotte = create_sentence_df(review, "Charlotte, NC", {"state": {"$in": ["NC","SC"]}, "categories": {"$in": restaurants}})


#Pickle Dataframes for Upload onto AWS for Further Analysis
madison.to_pickle("data/yelp/dataframes/review_sentences_madison.pkl")
pittsburg.to_pickle("data/yelp/dataframes/review_sentences_pittsburgh.pkl")
charlotte.to_pickle("data/yelp/dataframes/review_sentences_charlotte.pkl")









