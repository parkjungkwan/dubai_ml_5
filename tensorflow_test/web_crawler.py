from selenium import webdriver
from time import sleep
from bs4 import BeautifulSoup

class WebCrawler:
    def __init__(self):
        pass
    @staticmethod
    def create_model():
        context = './data/'
        driver = webdriver.Chrome(context + 'chromedriver')
        driver.get('https://movie.naver.com/movie/bi/mi/review.nhn?code=179158')
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        all_divs = soup.find_all('ul', attrs={'class', 'rvw_list_area'})
        products = [ul.li.p.a.string for ul in all_divs]
        driver.close()
        print('movies : {}'.format(products))
        return products