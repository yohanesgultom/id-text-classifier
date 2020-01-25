import requests
import pickle
import configparser
import sqlite3
import traceback
import argparse
from bs4 import BeautifulSoup
from train import (
    SimpleIndonesianPreprocessor,
    identity
)
from crawler import (
    DetikCrawler,
    KompasCrawler,
    Liputan6Crawler
)

CONFIG_FILE='.config'

config = configparser.ConfigParser()
config.readfp(open(CONFIG_FILE))

if __name__ == '__main__':    

    parser = argparse.ArgumentParser(description='Crawl and classify Indonesian commodity news')
    parser.add_argument('-s', '--silent', action='store_true', default=False, help='display no log in console')
    args = parser.parse_args()

    # crawl
    crawlers = [
        DetikCrawler.DetikCrawler(),
        KompasCrawler.KompasCrawler(),
        Liputan6Crawler.Liputan6Crawler(),
    ]

    titles = []
    links = []
    for c in crawlers:
        new_titles, new_links = c.crawl(args.silent)
        titles += new_titles
        links += new_links

    # classify titles
    MODEL_FILE = 'model.pkl'
    if not args.silent:
        print('Loading classifier {}..'.format(MODEL_FILE))

    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
        # force override verbosity
        model.set_params(preprocessor__verbose=not args.silent)

    X = titles
    y_predict = model.predict(X)

    # load db
    db = sqlite3.connect(config.get('APP', 'SQLITE3_DB'))
    cursor = db.cursor()

    # create table if not exists
    # make sure title is unique
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS titles(
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        link TEXT NOT NULL,
        crawled_at TEXT NOT NULL,
        posted_at TEXT,
        post_id TEXT,
        UNIQUE(title)
        )
    ''')
    db.commit()

    if not args.silent:
        print('Result:\n')

    count = 0
    for s, link, label in zip(X, links, y_predict):
        if not args.silent:
            print('{}\t{}'.format(s, label))
        # store new positive result to db    
        if label == '1':
            try:
                cursor.execute('''INSERT INTO 
                titles(title, link, crawled_at) 
                VALUES(?, ?, datetime('now', 'localtime'))
                ''', (s, link))
                count += 1
            except sqlite3.IntegrityError as e:
                # do nothing
                count = count
            except Exception as e:
                if not args.silent:
                    print(traceback.format_exc())
                    print('Error: Unable to store title and link: {} {}'.format(s, link))


    # commit insert
    db.commit()

    if not args.silent:
        print('\n{} new title(s) saved.'.format(count))

    # close db
    db.close()