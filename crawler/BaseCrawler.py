from abc import ABCMeta, abstractmethod

class BaseCrawler:
    __metaclass__ = ABCMeta

    @abstractmethod
    def crawl(self, silent=False): raise NotImplementedError
