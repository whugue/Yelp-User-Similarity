"""
Grid Search to Determine Optimal Linear SVM Parameters
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix


##Read in Binary Text Vectorizer
with open("binary_vectorizer.pkl") as f:
    binary_vectorizer = pickle.load(f)
    

##Read In and Clean ABSA Data - Create Binary Flags for Topics & Aggregate to Sentence Level
absa_data = pd.read_pickle("all_absa_data.pkl")

food = ["food","FOOD#QUALITY","FOOD#STYLE_OPTIONS","DRINKS#QUALITY","DRINKS#STYLE_OPTIONS","FOOD#GENERAL"]
service = ["service","SERVICE#GENERAL"]
ambience = ["ambience","AMBIENCE#GENERAL"]
value = ["price","FOOD#PRICES","RESTAURANT#PRICES","DRINKS#PRICES"]

topics = ["topic_food","topic_service","topic_ambience","topic_value"]
absa_data["topic_food"] = -1
absa_data["topic_service"] = -1
absa_data["topic_ambience"] = -1 
absa_data["topic_value"] = -1

absa_data.ix[absa_data.category.isin(food), "topic_food"] = 1
absa_data.ix[absa_data.category.isin(service), "topic_service"] = 1
absa_data.ix[absa_data.category.isin(ambience), "topic_ambience"] = 1
absa_data.ix[absa_data.category.isin(value), "topic_value"] = 1

absa_data = absa_data.groupby(by="sentence", as_index=False)[topics].max()


##Grid Search over Linear SVM For Optimal Parameters to Each Topic
def topic_grid_search(vectorizer, df, topic):
    grid={"C": [0.01,1,2]}
    
    df_X = vectorizer.transform(df["sentence"]).toarray()
    df_y = df[topic]
    
    clf = GridSearchCV(LinearSVC(), grid, cv=5).fit(df_X, df_y)
    
    print "Best Params for "+topic+":", clf.best_params_
    print "Accuracy: "+str(clf.best_score_)
    print ""


##Run Functions
topic_grid_search(binary_vectorizer, absa_data, "topic_food") 
topic_grid_search(binary_vectorizer, absa_data, "topic_service")
topic_grid_search(binary_vectorizer, absa_data, "topic_ambience")
topic_grid_search(binary_vectorizer, absa_data, "topic_value")

##Tune Model on Entire Dataset or Just Training Set?














