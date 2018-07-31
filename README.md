### Metis Final Project: Finding Similar Users on Online Review Sites
Large online review sites such as Yelp or TripAdvisor have tens or hundreds of millions of reviewers; as such, it can be difficult to find relevant reviews written by people who care about the same things you do. For my final Metis project, I combined supervised and unsupervised machine learning techniques to segment Yelp users based on topics most commonly discussed in their Yelp reviews.

You can read a full analysis on [my blog](http://www.huguedata.com/2016/07/15/yelp-me-out/), or check out the [presentation](https://github.com/whugue/Yelp-User-Similarity/blob/master/deck/Yelp-Final.pdf) I gave at Metis  Career Day on June 23, 2016.


#### Program Flow
The table below provides high-level overviews of what each analysis script does. More information (including specific input/ouput data) can be found in each script's header.

Program 	| Description | 
----------- | ----------- |
01-Preprocess-Yelp-Data.py | Clean raw Yelp data and split into sentences so that 1 record = 1 review sentence.
02-Parse-SemEval-Data.py | Parse raw SemEval data (XML format) into pandas dataframe.
03-Vectorizers.py | Fit binary, count, and TF-IDF vectorizers to entire vocubulary from SemEval and Yelp data.
04-Train-Classifiers.py | Train text classifiers on SemEval data and test performance.
05-Tune-Linear-SVM.py | Tune hyperparameters for best performing text classifier from (4) (Linear SVM).
06-Create-User-File.py | Predict topics in Yelp review sentences and aggregate to the user level.
07-Yelp-Cluster.ipynb | Cluster Yelp users based on topics most discussed in reviews.
08-Yelp-Viz.ipynb | Produce final visualizations for presentation.
