from abc import ABC, abstractmethod

class BaseCrawler(ABC):
    @abstractmethod
    def crawl(self, silent=False): raise NotImplementedError
