# Indonesian Text Classifer

Simple Indonesian text classifier using [Sklearn Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Two models are currently available:

1. TF-IDF vectorization with SGD Classifier (linear SVM) (default) [3]: 77% F1-score with small model size (< 800 KB)
2. Word2Vec x TF-IDF vectorization with RBF SVM [4]: 86% F1-score but with huge model size (> 1 GB) due to encapsulation of [FastText pretrained word vectors](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) (> 750 MB)

## Dependencies

* Python >= 2.7.x (not compatible yet with 3.x)
* Sqlite3

## Usage

### Training

1. Install dependencies `pip install -r requirements.txt`
1. Put datasets in `dataset_labeled.csv`
1. Run `python train.py`. Example:

```
$ python train.py -o model.pkl

Preprocessing..
100% (123 of 123) |####################################################################| Elapsed Time: 0:00:28 Time: 0:00:28Cross-validating..

No      test_pos_precision      test_pos_recall test_neg_precision      test_neg_recall
0       1.0     0.4     0.8696  1.0
1       1.0     1.0     1.0     1.0
2       1.0     0.6     0.9091  1.0
3       1.0     0.5     0.9091  1.0
4       1.0     0.5     0.9091  1.0

Avg     1.0     0.6     0.919367588933  1.0

Building complete model and saving ...
Preprocessing..
100% (123 of 123) |####################################################################| Elapsed Time: 0:00:00 Time: 0:00:00
Model saved in model.pkl
```

### Testing

1. Do training or obtain `model.pkl` (and put it in same location as `test.py`)
1. Install dependencies (if you have not) `pip install -r requirements.txt`
1. Run `python test.py "sentences_1" "sentences_2" "sentences_n"`. Example:

```
$ python test.py "Harga Gabah Jatuh karena Hujan Berkepanjangan" "Donatella Klaim Film Serial Pembunuhan Gianni Versace Fiktif"

Preprocessing..
100% (2 of 2) |########################################################################| Elapsed Time: 0:00:00 Time: 0:00:00
Prediction(s):
Harga Gabah Jatuh karena Hujan Berkepanjangan (1)
Donatella Klaim Film Serial Pembunuhan Gianni Versace Fiktif (0)
```

### Crawling titles from website

To automatically crawl titles (and links) from website, run classification and store positive results use `crawler.py`:

1. Copy `.config.example` to `.config`
1. Run `python crawler.py`


### Post crawled title links to facebook

To automatically post crawled links from website, use `fb.py`:

1. Copy `.config.example` to `.config` and replace all Facebook config with valid values
1. Run `python fb.py`

> To get token without expiry time follow suggestion from [documentation](https://developers.facebook.com/docs/facebook-login/access-tokens/expiration-and-extension) :
>
> "To get a longer-lived page access token, exchange the User access token for a long-lived one, as above, and then request the Page access token. The resulting page access token will not have any expiry time."

```
$ python crawl.py
```

## To Do List

* Replace stemming with lemmatizer
* Add more crawling sources
* Try reinforcement learning instead of supervised learning
* Migrate to Python 3.x

## References

1. Stemmer https://github.com/har07/PySastrawi
2. Stopwords list https://github.com/stopwords-iso/stopwords-id
3. Text classification with Sklearn Pipeline https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
4. Text classification with Sklearn and Gensim Word2Vec http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
5. Facebook FastText pretrained word vectors https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md