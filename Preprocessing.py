import re
import string

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd

datatweet = pd.read_csv('data_twitter.csv')

# Cleaning Data
def cleaning(tweet):
    regrex_pattern = re.compile(pattern='['
                                u'\U0001F600-\U0001F64F'
                                u'\U0001F300-\U0001F5FF'
                                u'\U0001F680-\U0001F6FF'
                                u'\U0001F1E0-\U0001F1FF'
                                ']+', flags=re.UNICODE)
    # Menghapus tanda baca, mengubah menjadi huruf kecil, menghapus spasi
    # Menghapus spasi di depan dan di belakang (whitespace leading dan trailing (strip))
    tweet = tweet.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    tweet = re.sub(r'@[A-Za-z0-9]', '', tweet)
    tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    tweet = re.sub(r'\n+', '', tweet) # Menghilangkan enter
    tweet = re.sub(r'\d+', '', tweet) # Menghilangkan angka
    tweet = regrex_pattern.sub(r'', tweet)
    return tweet

# Tokenizing
def tokenizing(tweet):
    tweet = nltk.tokenize.word_tokenize(tweet)
    return tweet

# Filtering/Stopword (nltk)
def filtering_nltk(tweet):
    liststopword = set(stopwords.words('indonesian'))
    removed = []
    for t in tweet:
        if t not in liststopword:
            removed.append(t)
    return removed

# Filtering/Stopword (sastrawi)
def filtering_sastrawi(tweet):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    stop = stopword.remove(tweet)
    return stop

def filtering_sastrawi_custom(tweet):
    stop_factory = StopWordRemoverFactory().get_stop_words()
    more_stopword = ['np']
    data = stop_factory + more_stopword
    dictionary = ArrayDictionary(data)
    stopwords = StopWordRemover(dictionary)
    stop = stopwords.remove(tweet)
    return stop

# Steaming nltk
def steaming1(tweet):
    hasil = []
    ps = PorterStemmer()
    for k in tweet:
        hasil.append(ps.stem(k))
    return hasil

# Steaming sastrawi
def steaming2(tweet):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    hasil = stemmer.stem(tweet)
    return hasil

# Preprocessing
df = datatweet['Tweet'].apply(cleaning)
df = df.apply(filtering_sastrawi)
df = df.apply(steaming2)
df = df.apply(tokenizing)
df = df.apply(steaming1)
print(df)
