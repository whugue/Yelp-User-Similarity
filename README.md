### Metis Project 5: Finding Similar Users on Online Review Sites

Large online review sites such as Yelp or TripAdvisor have tens or hundreds of millions of reviewers; as such, it can be difficult to find relevant reviews written by people who care about the same things you do. For my final Metis project, I combined supervised and unsupervised machine learning techniques to segment Yelp users based on topics most commonly discussed in their Yelp reviews.

You can read a description of this project on my blog [here](http://www.huguedata.com/2016/07/15/yelp-me-out/).


### Program Flow

N | Program | Purpose | Inputs | Outputs
_ | _______ | _______ | ______ | _______
1 | 01-Preprocess-Yelp-Data.py | Filter and Split | Business & Review MongoDB Collections | Lots of output and stuf: 1. as 2. a 3. list

First Header | Second Header
------------ | -------------
Content from cell 1 | Content from cell 2
Content in the first column | Content in the second column




### Code Dependencies
Numpy, Pandas, Sci-Kit Learn, Matplotlib, Seaborn, PyLab, Pickle
Xml, Json, Collections, NLTK, MongoDB and PyMongo, Warnings


### Data:
The analysis uses two data sources:

1. Restaurant review data collected by the International Workshop on Semantic Evaluation (SemEval), split out by sentence with each sentence labeled by topic (food, service, ambience, value, etc.). The raw SemEval data are saved in XML Files in the *data/SemEval/2014/* and *data/SemEval/2015/* folders.
Parsed and compiled SemEval data are saved as a pickled pandas dataframe here: *data/all_semeval_data.pkl*.

2. Yelp review data from the Yelp academic dataset. These data aren't included into this repo, but can be downloaded [here](https://www.yelp.com/dataset_challenge). Before running any of the below scripts, you'll need to [install mongoDB and PyMongo](https://docs.mongodb.com/manual/installation/) and load the Yelp JSON data into it using the following commands:

```
mongoimport --db yelp --collection business yelp_academic_dataset_business.json
mongoimport --db yelp --collection review yelp_academic_dataset_review.json
```


### Python Scripts, IPython Notebooks, and Program Flows:
1. *01-Break-Yelp-Into-Sentences.py*: Subset Yelp reviews into U.S. restaurant reviews and split into sentences (Split Yelp reviews out into sentences with one row/document per sentence.
2. *02-Parse-SemEval-Data.py*: Parse raw SemEval data (XML format) into pandas dataframe.
3. *03-Vectorizers.py*: Fit binary, count, and TF-IDF vectorizers to entire vocubulary from SemEval and Yelp data.
4. *04-Train-Classifiers.py*: Train text classifiers on SemEval data and test performance.
5. *05-Tune-Linear-SVM.py*: Tune hyperparameters for best performing text classifier from (4) (Linear SVM).
6. *06-Create-User-File.py*: Predict topics in Yelp review sentences and aggregate to the user level.
7. *07-Yelp-Cluster.ipynb*: Cluster Yelp users based on topics most discussed in reviews.
8. *08-Yelp-Viz.ipynb*: Produce final visualizations for presentation.


### Presentation:
*deck/*: Deck from final presentation given at Metis on 6/23/2016.