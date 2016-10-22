"""
Script:     04-Train-Classifiers.py
Purpose:    Train various supervised text clasisifcaiton models and determine best overall performer
Input:      data/SemEval/all_semeval_data.pkl
            data/vectorizers/binary_vectorizer.pkl
            data/vectorizers/count_vectorizer.pkl //TODO: Save to GH Repo
            data/vectorizers/tfidf_vectorizer.pkl //TODO: Save to GH Repo
Output:     Various performance statistics // TODO: Output these to some sort of graphic, not just as printed output 
"""


import numpy as np
import pandas as pd
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix


##Read in SemEval Data
semeval_data = pd.read_pickle("data/SemEval/all_semeval_data.pkl")
#semeval_data.category.value_counts(dropna=False)


##Clean SemEval Data - Create Binary Flags for Topics & Aggregate to Sentence Levle
food = ["food","FOOD#QUALITY","FOOD#STYLE_OPTIONS","DRINKS#QUALITY","DRINKS#STYLE_OPTIONS","FOOD#GENERAL"]
service = ["service","SERVICE#GENERAL"]
ambience = ["ambience","AMBIENCE#GENERAL"]
value = ["price","FOOD#PRICES","RESTAURANT#PRICES","DRINKS#PRICES"]

topics = ["topic_food","topic_service","topic_ambience","topic_value"]
semeval_data["topic_food"] = 0
semeval_data["topic_service"] = 0
semeval_data["topic_ambience"] = 0 
semeval_data["topic_value"] = 0

semeval_data.ix[semeval_data.category.isin(food), "topic_food"] = 1
semeval_data.ix[semeval_data.category.isin(service), "topic_service"] = 1
semeval_data.ix[semeval_data.category.isin(ambience), "topic_ambience"] = 1
semeval_data.ix[semeval_data.category.isin(value), "topic_value"] = 1

semeval_data = semeval_data.groupby(by="sentence", as_index=False)[topics].max()


##Split Data innto Train and Test
semeval_train, semeval_test = train_test_split(semeval_data, test_size=0.25, random_state=4444)
#print semeval_train.shape
#print semeval_test.shape



"""
Function to read in vectorizers

inpickle:   Path to pickled vectorizer
RETURNS:    Deserialized vectorizer
"""
def open_vectorizer(inpickle):
    with open(inpickle) as f:
        vectorizer = pickle.load(f)

    return vectorizer

binary_vectorizer = open_vectorizer("data/vectorizers/binary_vectorizer.pkl")
count_vectorizer = open_vectorizer("data/vectorizers/count_vectorizer.pkl")
tfidf_vectorizer = open_vectorizer("data/vectorizers/tfidf_vectorizer.pkl")



"""
Function to train text classifiers

vectorizer:     Vectorizer to use in model training
classifier:     Classification algorithm to train (e.g. NB, SVM)
train:          Training data (pandas DF)
test:           Validation data (pandsd DF)
topic:          Topic to train model to identify (Food, Service, Ambience, Value)

RETURNS:        Print of Accuracy & Confusion Matrix for specific classifcation algorithm
"""
def train_clf(vectorizer, classifier, train, test, topic):    
    train_X = vectorizer.transform(train["sentence"]).toarray()     #Vectorize Training Features
    test_X = vectorizer.transform(test["sentence"]).toarray()       #Vectorize Testing Feature 
    
    train_y = train[topic]                                          #Create Training Label Vector
    test_y = test[topic]                                            #Create Testing Label Vector
    
    dummy_clf = DummyClassifier(strategy="most_frequent").fit(train_X, train_y) #Train a Dummy Classifier (for comparison)
    clf = classifier.fit(train_X, train_y)                                      #Train Actual Classifier
    

    #Test Classifiers & Output Accuracy, Confusion Matrix Statistics
    dummy_accuracy = accuracy_score(test_y, dummy_clf.predict(test_X))
    accuracy = accuracy_score(test_y, clf.predict(test_X))
    cm = confusion_matrix(test_y, clf.predict(test_X))
    
    print topic+" Dummy Accuracy: "+str(dummy_accuracy)
    print topic+" Accuracy:       "+str(accuracy)
    print topic+" Confusion Matrix: "
    print cm
    print ""
    

##Iterate Through All Topic Spaces
print "Training Bernoulli Naive Bayes Classifier..."
train_clf(binary_vectorizer, BernoulliNB(binarize=None), semeval_train, semeval_test, "topic_food")
train_clf(binary_vectorizer, BernoulliNB(binarize=None), semeval_train, semeval_test, "topic_service")
train_clf(binary_vectorizer, BernoulliNB(binarize=None), semeval_train, semeval_test, "topic_ambience")
train_clf(binary_vectorizer, BernoulliNB(binarize=None), semeval_train, semeval_test, "topic_value")


print "Training Multinomial Naive Bayes Classifier..."
clf_food = train_clf(count_vectorizer, MultinomialNB(), semeval_train, semeval_test, "topic_food")
clf_service = train_clf(count_vectorizer, MultinomialNB(), semeval_train, semeval_test, "topic_service")
clf_ambience = train_clf(count_vectorizer, MultinomialNB(), semeval_train, semeval_test, "topic_ambience")
clf_value = train_clf(count_vectorizer, MultinomialNB(), semeval_train, semeval_test, "topic_value")


print "Training Random Forest Classifier..."
clf_food = train_clf(binary_vectorizer, RandomForestClassifier(), semeval_train, semeval_test, "topic_food")
clf_service = train_clf(binary_vectorizer, RandomForestClassifier(), semeval_train, semeval_test, "topic_service")
clf_ambience = train_clf(binary_vectorizer, RandomForestClassifier(), semeval_train, semeval_test, "topic_ambience")
clf_value = train_clf(binary_vectorizer, RandomForestClassifier(), semeval_train, semeval_test, "topic_value")


print "Training Linear SVM..."
clf_food = train_clf(binary_vectorizer, LinearSVC(), semeval_train, semeval_test, "topic_food")
clf_service = train_clf(binary_vectorizer, LinearSVC(), semeval_train, semeval_test, "topic_service")
clf_ambience = train_clf(binary_vectorizer, LinearSVC(), semeval_train, semeval_test, "topic_ambience")
clf_value = train_clf(binary_vectorizer, LinearSVC(), semeval_train, semeval_test, "topic_value")


"""
Not doing these, they take forever/ error out due to lack of memory (even on EC2) and have good performance w/ Linear SVM
print "Training Gradient Boosting Classifier..."
clf_food = train_clf(binary_vectorizer, GradientBoostingClassifier(), semeval_train, semeval_test, "topic_food")
clf_service = train_clf(binary_vectorizer, GradientBoostingClassifier(), semeval_train, semeval_test, "topic_service")
clf_ambience = train_clf(binary_vectorizer, GradientBoostingClassifier(), semeval_train, semeval_test, "topic_ambience")
clf_value = train_clf(binary_vectorizer, GradientBoostingClassifier(), semeval_train, semeval_test, "topic_value")


print "Training Nonlinear SVM..."
clf_food = train_clf(binary_vectorizer, SVC(), semeval_train, semeval_test, "topic_food")
clf_service = train_clf(binary_vectorizer, SVC(), semeval_train, semeval_test, "topic_service")
clf_ambience = train_clf(binary_vectorizer, SVC(), semeval_train, semeval_test, "topic_ambience")
clf_value = train_clf(binary_vectorizer, SVC(), semeval_train, semeval_test, "topic_value")
"""


