### Metis Project 5: Finding Similar Users on Online Review Sites

Large online review sites such as Yelp or TripAdvisor have tens or hundreds of millions of reviewers; as such, it can be difficult to find relevant reviews written by people who care about the same things you do. For my final Metis project, I combined supervised and unsupervised machine learning techniques to 

You can read a description of this project on my blog [here](http://www.huguedata.com/2016/04/28/oscar-bait-a-scientific-investigation/)


### Code Dependencies
* Numpy
* Pandas
* Sklearn
* Matplotlib
* Seaborn
* PyLab
* Pickle
* Xml
* Json
* Bson
* Collections
* NLTK
* MongoDB and PyMongo
* Warnings


### Data:
The analysis uses two data sources:
1. Restaurant review data collected by the International Workshop on Semantic Evaluation (SemEval), split out by sentence with each sentence labeled by topic (food, service, ambience, value, etc.). Parsed IWSE data is saved as a pandas dataframe here: ("/deck")

2. Yelp review data from the Yelp academic dataset. These data aren't included into this repo, but can be downloaded [here](https://www.yelp.com/dataset_challenge). Before running any of the below scripts, you'll need to download these data and load the business and review JSON files into a mongoDB database as two collection named "buisness" and "review"


### Python Scripts, IPython Notebooks, and Program Flows:
1. *break_yelp_into_sentences.py*: Subset Yelp reviews into U.S. restaurant reviews and split into sentences (Split Yelp reviews out into sentences with one row/document per sentence.
2. *parse_absa_data.py*: Parse raw IWSE data (XML format) into pandas dataframe.
3. *vectorizers.py*: Fit binary, count, and TF-IDF vectorizers to entire vocubulary from IWSE and Yelp data.
4. *train_classifiers.py*: Train text classifiers on IWSE data and test performance.
5. *tune_linear_svm.py*: Tune hyperparameters for best performing text classifier from (4) (Linear SVM).
6. *create_user_file.py*: Predict topics in Yelp review sentences and aggregate to the user level.
7. *Yelp-Cluster.ipynb*: Cluster Yelp users based on topics most discussed in reviews.
8. *Yelp-Viz.ipynb*: Produce final visualizations for presentation.


### Presentation:
*deck/*: Deck from final presentation given at Metis on 6/23/2016