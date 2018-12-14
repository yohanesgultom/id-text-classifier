"""
Detik news titles and links crawler
@Author yohanes.gultom@gmail.com
"""

import requests
import traceback
from BaseCrawler import BaseCrawler
from bs4 import BeautifulSoup

class DetikCrawler(BaseCrawler):
    def __init__(self):
        self.url = 'https://finance.detik.com/ekonomi-bisnis/indeks'
        self.list_selector = 'article a'

    @staticmethod
    def __crawl(url, list_selector, silent=False):
        if not silent:
            print('Crawling {} ..'.format(url))
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        tags = soup.select(list_selector)
        titles = []
        links = []
        for a in tags:
            try:
                # remove unicode chars and whitelines
                titles.append(a.text.encode('utf-8').strip())
                links.append(a['href'])
            except Exception as e:
                if not silent:
                    print(traceback.format_exc())
                    print('Error: Unable to process anchor: {}'.format(a))
        return titles, links, soup

    def crawl(self, silent=False):
        titles, links, soup = self.__crawl(self.url, self.list_selector, silent)
        # no pagination
        return titles, links

        
    
if __name__ == '__main__':
    titles, links = DetikCrawler().crawl()
    for t, l in zip(titles, links):
        print('{} {}'.format(t, l))
    