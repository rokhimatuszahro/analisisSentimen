import pandas as pd
# Import modul reglar expression
import re
import string

import nltk
# Import kelas word_tokenize
from nltk.tokenize import word_tokenize
# Import kelas FreqDist
from nltk.probability import FreqDist
# Import kelas PosterStemmer
from nltk.stem import PorterStemmer
# Import kelas stopwords
from nltk.corpus import stopwords

# Import kelas StemmerFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Membaca data Twitter yang berupa file .csv menggunakan modul pandas
dataTweet = pd.read_csv('data_twitter.csv')
print(dataTweet.head(10))

#---------------------------------------- Cleaning Data ----------------------------------------#
# Cleaning data yaitu proses dimana menghilangkan noise pada data agar tidak mengganggu pada saat pemrosesan data.
def cleaning(tweet):
    # Menghapus emoticon
    regrex_pattern = re.compile(pattern='['
                                        u'\U0001F600-\U0001F64F'
                                        u'\U0001F300-\U0001F5FF'
                                        u'\U0001F680-\U0001F6FF'
                                        u'\U0001F1E0-\U0001F1FF'
                                        ']+', flags=re.UNICODE)
    # Menghapus tanda baca, mengubah huruf kapital menjadi huruf kecil semua, menghapus whitespace (karakter kosong)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    # Menghapus Mention
    tweet = re.sub(r'@[A-Za-z0-9]', '', tweet)
    # Menghapus link
    tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
    # Menghapus hashtag
    tweet = re.sub(r'#', '', tweet)
    # Menghapus url
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    # Menghilangkan beberapa spasi menjadi satu spasi (multiple whitespace into single whitespace)
    tweet = re.sub('\s+', ' ', tweet)
    # Menghapus karakter tunggal (single char)
    tweet = re.sub(r"\b[a-zA-Z]\b", "", tweet)
    # Menghapus angka
    tweet = re.sub(r'\d+', '', tweet)
    #
    tweet = regrex_pattern.sub(r'', tweet)
    # Menghilangkan tab, baris baru, dan back slice
    tweet = tweet.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    # Menghilangkan non ASCII (emoticon, chinese word, .etc)
    tweet = tweet.encode('ascii', 'replace').decode('ascii')
    return tweet


#---------------------------------------- Tokenizing ----------------------------------------#
# Tokenizing proses pemisahan teks menjadi potongan-potongan kata yang disebut token untuk kemudian dianalisa.
def tokenizing(tweet):
    tweet = nltk.tokenize.word_tokenize(tweet)
    return tweet

# Menghitung frekuensi distribusi token pada tiap row data pada Dataframe
def freqdist(tweet):
    return FreqDist(tweet)

# print(dataTweet.head().apply(lambda x : x.most_common()))


#---------------------------------------- Normalisasi ----------------------------------------#
# Membaca data normalisasi yang berupa file .xlsx menggunakan modul pandas
normalizad_word = pd.read_excel("Book1.xlsx")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]

def normlized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]


#---------------------------------------- Filtering (Stopword Removel) ----------------------------------------#
# Filtering menggunakan nltk
def filtering_nltk(tweet):
    # Mendapatkan stopwords indonesia
    liststopword = set(stopwords.words('indonesian'))
    removed = []
    for t in tweet:
        if t not in liststopword:
            removed.append(t)
    return removed

# Filtering menggunakan sastrawi
def filtering_sastrawi(tweet):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    stop = stopword.remove(tweet)
    return stop

# Filtering menggunakan sastrawi (custom)
def filtering_sastrawi_custom(tweet):
    stop_factory = StopWordRemoverFactory().get_stop_words()
    more_stopword = ['np']
    data = stop_factory + more_stopword
    dictionary = ArrayDictionary(data)
    stopwords = StopWordRemover(dictionary)
    stop = stopwords.remove(tweet)
    return stop


#---------------------------------------- Steaming ----------------------------------------#
# Steaming menggunakan nltk
def steaming_nltk(tweet):
    hasil = []
    ps = PorterStemmer()
    for k in tweet:
        hasil.append(ps.stem(k))
    return hasil

# Steaming menggunakan sastrawi
def steaming_sastrawi(tweet):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    hasil = stemmer.stem(tweet)
    return hasil



#---------------------------------------- PREPROCESSING ----------------------------------------#
dataTweet['cleaning'] = dataTweet['tweet'].apply(cleaning)
dataTweet['tokenizing'] = dataTweet['cleaning'].apply(tokenizing)
dataTweet['normalisasi'] = dataTweet['tokenizing'].apply(normlized_term)
dataTweet['filtering_sastrawi'] = dataTweet['normalisasi'].apply(filtering_sastrawi)
dataTweet['steaming_sastrawi'] = dataTweet['filtering_sastrawi'].apply(steaming_sastrawi)
dataTweet['steaming_nltk'] = dataTweet['steaming_sastrawi'].apply(steaming_nltk)
print(dataTweet['steaming_nltk'].head(10))
