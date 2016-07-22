"""
(IV) Test Various Topic Text Classification Models to Determine Best Performer:
    (1) Bernoulli Naive Bayes
    (2) Multinomial Naive Bayes
    (3) Random Forest
    (4) Linear SVM
    (5) Gradient Boosted Trees (Didn't Run - Took too long and already had performance needed from simpler classifiers)
    (6) Nonlinear SVM (Didn't Run - Took too long and already had performance needed from simpler classifiers)
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

##Read in ABSA Data
semeval_data = pd.read_pickle("all_semeval_data.pkl")
#semeval_data.category.value_counts(dropna=False)


##Clean ABSA Data - Create Binary Flags for Topics & Aggregate to Sentence Levle
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


##Read in Vectorizers
def open_vectorizer(inpickle):
    with open(inpickle) as f:
        vectorizer = pickle.load(f)

    return vectorizer

binary_vectorizer = open_vectorizer("binary_vectorizer.pkl")
count_vectorizer = open_vectorizer("count_vectorizer.pkl")
tfidf_vectorizer = open_vectorizer("tfidf_vectorizer.pkl")


##Train a Specific Classifier Based on a Specific Vectoriation
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
Not doing these, they take forever/ error out due to lack of memory, even on EC2
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


