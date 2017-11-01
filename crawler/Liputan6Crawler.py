"""
Liputan 6 news titles and links crawler
@Author yohanes.gultom@gmail.com
"""

import requests
import traceback
from BaseCrawler import BaseCrawler
from bs4 import BeautifulSoup

class Liputan6Crawler(BaseCrawler):
    def __init__(self):
        self.url = 'http://www.liputan6.com/tag/pasar-tradisional'
        self.list_selector = '.articles--iridescent-list--text-item__title-link'
        self.attrib = 'href'
        self.pagination_class = 'simple-pagination__page-number-link'
        self.pagination_class_active = 'simple-pagination__page-number-link_active'

    @staticmethod
    def __crawl(url, list_selector, silent=False):
        if not silent:
            print('Crawling {} ..'.format(url))
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        # get only top 5 because the rests are usually too old
        limit = 5        
        tags = soup.select(list_selector)[:limit]
        titles = []
        links = []
        for a in tags:
            try:
                # remove unicode chars and whitelines
                titles.append(a.text.strip().encode('ascii', errors='ignore').encode('utf-8'))
                links.append(a['href'])
            except Exception as e:
                if not silent:
                    print(traceback.format_exc())
                    print('Error: Unable to process anchor: {}'.format(a))
        return titles, links, soup

    def crawl(self, silent=False):
        titles, links, soup = self.__crawl(self.url, self.list_selector, silent)
        # ignore pagination as the news are too old
        # pagination = soup.find_all(class_=self.pagination_class)
        # next_urls = set([a[self.attrib] for a in pagination if self.pagination_class_active not in a['class']])
        # for next_url in next_urls:
        #     next_titles, new_links, _ = self.__crawl(next_url, self.list_selector, silent)
        #     titles += next_titles
        #     links += new_links

        return titles, links

        
    
if __name__ == '__main__':
    titles, links = Liputan6Crawler().crawl()
    for t, l in zip(titles, links):
        print('{} {}'.format(t, l))
    