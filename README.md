# Indonesian Text Classifer

Simple Indonesian text classifier using [Sklearn Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Two models are currently available:

1. TF-IDF vectorization with SGD Classifier (linear SVM) (default) [3]: 61% F1-score with small model size (< 800 KB)
2. Word2Vec x TF-IDF vectorization with RBF SVM [4]: 77% F1-score but with huge model size (> 1 GB) due to encapsulation of [FastText pretrained word vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.id.vec) (> 750 MB)

## Dependencies

* Python >= 3.x
* Sqlite3

## Usage

### Training

1. Install dependencies `pip install -r requirements.txt`
1. Put datasets in `dataset_labeled.csv`
1. Run `python train.py`. Example:

```
$ python train.py

Cross-validating..
Classifier: SGDClassifier

No      test_pos_f1     test_pos_precision      test_pos_recall test_neg_f1     test_neg_precision      test_neg_recall
0       0.625   1.0     0.4545  0.9     0.8182  1.0
1       0.7778  1.0     0.6364  0.931   0.871   1.0
2       0.6667  0.8571  0.5455  0.8966  0.8387  0.963
3       0.5333  1.0     0.3636  0.8852  0.7941  1.0
4       0.625   1.0     0.4545  0.9     0.8182  1.0
5       0.8182  0.8182  0.8182  0.9259  0.9259  0.9259
6       0.7619  0.8889  0.6667  0.9091  0.8621  0.9615
7       0.1333  0.25    0.0909  0.7797  0.697   0.8846

Avg     0.6176  0.8518  0.5038  0.8909  0.8281  0.9669
```

### Training using Word2Vec

1. Download Indonesian Wikipedia word2vec model from Facebook Research [wiki.id.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.id.vec) and put it in the same directory as `train.py`
1. Put datasets in `dataset_labeled.csv`
1. Run `python train.py -m w2v-rbfsvm`
1. Run `python train.py -m w2v-rbfsvm -e dataset_labeled.vec` to use custom embedding file trained using `word2vec.py`

```
Cross-validating..
Classifier: SVC

No      test_pos_f1     test_pos_precision      test_pos_recall test_neg_f1     test_neg_precision      test_neg_recall
0       0.8421  1.0     0.7273  0.9474  0.9     1.0
1       0.8421  1.0     0.7273  0.9474  0.9     1.0
2       0.8     0.8889  0.7273  0.9286  0.8966  0.963
3       0.625   1.0     0.4545  0.9     0.8182  1.0
4       0.9     1.0     0.8182  0.9643  0.931   1.0
5       0.8696  0.8333  0.9091  0.9434  0.9615  0.9259
6       0.8696  0.9091  0.8333  0.9434  0.9259  0.9615
7       0.4286  1.0     0.2727  0.8667  0.7647  1.0

Avg     0.7721  0.9539  0.6837  0.9301  0.8872  0.9813
```

> You can train your own word2vec using `python word2vec.py` by putting training data (sentences) in first column of `dataset_labeled.csv`

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

* Store word2vec model in database to save RAM
* Replace stemming with lemmatizer
* Add more crawling sources
* Try reinforcement learning instead of supervised learning

## References

1. Stemmer https://github.com/har07/PySastrawi
2. Stopwords list https://github.com/stopwords-iso/stopwords-id
3. Text classification with Sklearn Pipeline https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
4. Text classification with Sklearn and Gensim Word2Vec http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
5. Facebook FastText pretrained word vectors https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md