
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
location:       Location of Yelp Businessed pulled from MongoDB
query:          Mongo query to filter all Yelp businesses to ONLY restauarants in a given city
RETURNS:        List of Business IDs for restaurants in a given city (as specified by input mongo_query)
"""
def get_business_ids(query):
    print "Filtering Businesses from: ", location
    business_ids = []
    cursor = business.find(mongo_query)

    for each in cursor:
        business_ids.append(each["business_id"])

    return business_ids


"""
Function to break each review into it's compotnent sentences
location:       Location of Yelp Businessed pulled from MongoDB
business_ids:   List of Yelp ID for restaurants in given city pulled from MongoDB

RETURNS:        Pandas DF of restaurant reviews for given location, where 1 row = 1 sentence
"""
def create_sentence_df(location, business_ids):
    print "Creating Cursor for ", location, "..."
    cursor = review.find({"business_id": {"$in": business_ids}})

    #Split Each Review into it's component sentences with one row = one review sentence
    print "Spliting Reviews into Sentences for ", location, "..."

    df = []
    for review in cursor:
        sentences = sent_tokenize(review["text"])

        for sentence in sentences:
            row=defaultdict(str)

            row["user_id"] = review["user_id"]  #User ID
            row["review_id"] = review["_id"]    #Review ID (Object ID from MongoDB)
            row["location"] = location          #Location (from function input)
            row["sentence"] = sentence

            df.append(row)

    print "DF Created for: ", location
    return pd.DataFrame(df)



##Run Above Function to Filter & Clean Yelp Review Data for Analysis
##This JSON file contains a list of all Yelp business categories considered as "restaurants for this analysis."
##These categories were derived by hand from a list of all possible business categories in the raw Yelp data
restaurants = json.loads(open("data/yelp/restaurants.json","r+").read())["food_places"]


##Filter to Resturant Reviews for [city] and split out into component sentences
madison_business_ids = get_business_ids({"state":"WI", "categories": {"$in": restaurants}})
pittsburg_business_ids = get_business_ids({"state": "PA", "categories": {"$in": restaurants}})
charlotte_busines_ids = get_business_ids({"state": {"$in": ["NC","SC"]}, "categories": {"$in": restaurants}})


##Split Reviews out into component sentences
madison = create_sentence_df("Madison, WI", madison_business_ids)                       
pittsburg = create_sentence_df("Pittsburg, PA", pittsburg_business_ids)                 
charlotte = create_sentence_df("Charlotte, NC", charlotte_busines_ids)


#Pickle Dataframes for Upload onto AWS for Further Analysis
madison.to_pickle("data/yelp/dataframes/review_sentences_madison.pkl")
pittsburg.to_pickle("data/yelp/dataframes/review_sentences_pittsburgh.pkl")
charlotte.to_pickle("data/yelp/dataframes/review_sentences_charlotte.pkl")









