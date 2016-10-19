### Data:
The analysis uses two data sources:

**NOTE**: Please note that the Yelp data are not saved to this repo (they are a litle too big :)). However, the raw Yelp data are easily downloaded (for FREE!) [here]((https://www.yelp.com/dataset_challenge). Processed Yelp data can be reproduced by running the following python scripts:

1. 01-Preprocess-Yelp-Data.py
2. 07-Create-User-File.py

Before running the first script, be sure to load the Yelp data to a MongoDB on your local machine. After [installing MongoDB](https://docs.mongodb.com/manual/installation/), you can load the JSON Yelp Academic Dataset onto Mongo using the following command line prompts:

```
mongoimport --db yelp --collection business yelp_academic_dataset_business.json
mongoimport --db yelp --collection review yelp_academic_dataset_review.json
```

