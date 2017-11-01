"""
Kompas news titles and links crawler
@Author yohanes.gultom@gmail.com
"""

import requests
import traceback
from BaseCrawler import BaseCrawler
from bs4 import BeautifulSoup

class KompasCrawler(BaseCrawler):
    def __init__(self):
        self.url = 'http://indeks.kompas.com/ekonomi/'
        self.list_selector = '.article__link'
        self.attrib = 'href'
        self.pagination_selector = '.paging__link'
        self.excludes = ('#', 'javascript:')

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
        pagination = soup.select(self.pagination_selector)
        next_urls = set([a[self.attrib] for a in pagination if a[self.attrib].lower() not in self.excludes])
        for next_url in next_urls:
            next_titles, new_links, _ = self.__crawl(next_url, self.list_selector)
            titles += next_titles
            links += new_links

        return titles, links

        
    
if __name__ == '__main__':
    titles, links = KompasCrawler().crawl()
    for t, l in zip(titles, links):
        print('{} {}'.format(t, l))
    