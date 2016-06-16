

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

##Read in ABSA Data
absa_data = pd.read_pickle("all_absa_data.pkl")
#absa_data.category.value_counts(dropna=False)


##Clean ABSA Data - Create Binary Flags for Topics & Aggregate to Sentence Levle
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


##Split Data innto Train and Test
absa_train, absa_test = train_test_split(absa_data, test_size=0.25, random_state=4444)
#print absa_train.shape
#print absa_test.shape


##Read in Vectorizers
def open_vectorizer(inpickle):
    with open(inpickle) as f:
        vectorizer = pickle.load(f)

    return vectorizer

binary_vectorizer = open_vectorizer("binary_vectorizer.pkl")
count_vectorizer = open_vectorizer("count_vectorizer.pkl")
tfidf_vectorizer = open_vectorizer("tfidf_vectorizer.pkl")


##Train a Specific Classifier Based on a Specific Vectoriation
def train_clf(vectorizer, classifier, train, test, topic, clfpickle):    
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
    
    #Pickle Trained Classifiers
    with open(clfpickle, "wb") as f:
        pickle.dump(clf, f)


##Iterate Through All Topic Spaces
"""
print "Training Bernoulli Naive Bayes Classifier..."
train_clf(binary_vectorizer, BernoulliNB(binarize=None), absa_train, absa_test, "topic_food", "clf_bnb_food.pkl")
train_clf(binary_vectorizer, BernoulliNB(binarize=None), absa_train, absa_test, "topic_service", "clf_bnb_services.pkl")
train_clf(binary_vectorizer, BernoulliNB(binarize=None), absa_train, absa_test, "topic_ambience", "clf_bnb_ambience.pkl")
train_clf(binary_vectorizer, BernoulliNB(binarize=None), absa_train, absa_test, "topic_value", "clf_bnb_value.pkl")

print "Training Multinomial Naive Bayes Classifier..."
clf_food = train_clf(count_vectorizer, MultinomialNB(), absa_train, absa_test, "topic_food", "clf_mnb_food.pkl")
clf_service = train_clf(count_vectorizer, MultinomialNB(), absa_train, absa_test, "topic_service", "clf_mnb_service.pkl")
clf_ambience = train_clf(count_vectorizer, MultinomialNB(), absa_train, absa_test, "topic_ambience", "clf_mnb_ambience.pkl")
clf_value = train_clf(count_vectorizer, MultinomialNB(), absa_train, absa_test, "topic_value", "clf_mnb_value.pkl")

print "Training Random Forest Classifier..."
clf_food = train_clf(binary_vectorizer, RandomForestClassifier(), absa_train, absa_test, "topic_food", "clf_rf_food.pkl")
clf_service = train_clf(binary_vectorizer, RandomForestClassifier(), absa_train, absa_test, "topic_service", "clf_rf_service.pkl")
clf_ambience = train_clf(binary_vectorizer, RandomForestClassifier(), absa_train, absa_test, "topic_ambience", "clf_rf_ambience.pkl")
clf_value = train_clf(binary_vectorizer, RandomForestClassifier(), absa_train, absa_test, "topic_value", "clf_rf_value.pkl")
"""

print "Training Linear SVM..."
clf_food = train_clf(binary_vectorizer, LinearSVC(), absa_train, absa_test, "topic_food", "clf_linsvm_food.pkl")
clf_service = train_clf(binary_vectorizer, LinearSVC(), absa_train, absa_test, "topic_service", "clf_linsvm_service.pkl")
clf_ambience = train_clf(binary_vectorizer, LinearSVC(), absa_train, absa_test, "topic_ambience", "clf_linsvm_ambience.pkl")
clf_value = train_clf(binary_vectorizer, LinearSVC(), absa_train, absa_test, "topic_value", "clf_linsvm_value.pkl")


""" Not doing these, they take forever/ error out due to lack of memory, even on EC2
print "Training Gradient Boosting Classifier..."
clf_food = train_clf(binary_vectorizer, GradientBoostingClassifier(), absa_train, absa_test, "topic_food", "clf_gbt_food.pkl")
clf_service = train_clf(binary_vectorizer, GradientBoostingClassifier(), absa_train, absa_test, "topic_service","clf_gbt_service.pkl")
clf_ambience = train_clf(binary_vectorizer, GradientBoostingClassifier(), absa_train, absa_test, "topic_ambience", "clf_gbt_ambience.pkl")
clf_value = train_clf(binary_vectorizer, GradientBoostingClassifier(), absa_train, absa_test, "topic_value", "clf_gbt_value.pkl")


print "Training Nonlinear SVM..."
clf_food = train_clf(binary_vectorizer, SVC(), absa_train, absa_test, "topic_food", "clf_nonsvm_food.pkl")
clf_service = train_clf(binary_vectorizer, SVC(), absa_train, absa_test, "topic_service", "clf_nonsvm_service.pkl")
clf_ambience = train_clf(binary_vectorizer, SVC(), absa_train, absa_test, "topic_ambience", "clf_nonsvm_ambience.pkl")
clf_value = train_clf(binary_vectorizer, SVC(), absa_train, absa_test, "topic_value", "clf_nonsvm_value.pkl")
"""


