from tensorflow_test.mammal import Mammal
from tensorflow_test.word_sequence import WordSequence
from tensorflow_test.naive_bayes import NaiveBayes
from tensorflow_test.web_crawler import WebCrawler
if __name__ == '__main__':
    # Mammal.execute()
    # WordSequence.execute()
    # t = WebCrawler.create_model()
    nb = NaiveBayes()
    nb.train('./data/review_train.csv')
    print(nb.classify('내 인생에서 최고의 영화'))



