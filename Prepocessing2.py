import pandas as pd
# import numpy as np
import string
import re # regex library
import swifter
import nltk
# nltk.download()
import Sastrawi

# import word_tokeniize dan FreqDist dari NLTK
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import  word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Membaca file csv menggunakan pandas
dataTweet = pd.read_csv("data_twitter.csv")
# print(dataTweet.head())


#--- Case Folding ---#
# Gunakan fungsi Series.str.lower() pada Pandas
dataTweet['Tweet'] = dataTweet['Tweet'].str.lower()

# print('Hasil proses Case Folding: \n')
# print(dataTweet['Tweet'].head(10))


#--- Tokenizing ---#
def remove_tweet_special(text):
    # Menghilangkan tab, baris baru, ans back slice
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    # Menghilangkan non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # Menghilangkan mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # Menghilangkan incomplete URL
    return text.replace("http://", " ").replace("http://", " ")

# Menghilangkan angka
def remove_number(text):
    return re.sub(r"\d+", "", text)

# Menghilangkan tanda baca
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))


# Menghapus spasi di depan dan di belakang (whitespace leading dan trailing)
def remove_whitespace_LT(text):
    return text.strip()

# Menghapus beberapa spasi menjadi satu spasi (multiple whitespace into single whitespace)
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ', text)

# Menghapus karakter tunggal (single char)
def remove_single_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

# NLTK word tokenize
def word_tokenize_wrapper(text):
    return word_tokenize(text)

proses_1 = dataTweet['Tweet'].apply(remove_tweet_special)
proses_2 = proses_1.apply(remove_number)
proses_3 = proses_2.apply(remove_punctuation)
proses_4 = proses_3.apply(remove_whitespace_LT)
proses_5 = proses_4.apply(remove_whitespace_multiple)
proses_6 = proses_5.apply(remove_single_char)
dataTweet['Tweet_tokens'] = proses_6.apply(word_tokenize_wrapper)
# print(dataTweet['Tweet_tokens'].head(10))

# Menghitung frekuensi distribusi token pada tiap row data pada Dataframe
def freqDist_wrapper(text):
    return FreqDist(text)

dataTweet['Tweet_tokens_fdist'] = dataTweet['Tweet_tokens'].apply(freqDist_wrapper)
# print('Frequency Tokens: \n')
# print(dataTweet['Tweet_tokens_fdist'].head().apply(lambda x : x.most_common()))


#--- Filtering (Stopword Removal) ---#
# Mendapatkan stopword dari NLTK stopword
# Mendapatkan stopword indonesia
list_stopwords = stopwords.words('indonesian')

# Menambahkan stopword secara manual
# Tambahkan stopword tambahan
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                       'kalo', 'amp', 'biar', 'bikin', 'bilang',
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# Tambahkan stopword dari file txt
# Baca stopword txt menggunakan pandas
# txt_stopword = pd.read_csv("stopwords.txt", names= ["stopwords"], header = None)

# Konversikan string stopword ke daftar & tambahkan stopword tambahan
# list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

# Mengubah daftar ke kamus
list_stopwords = set(list_stopwords)

# Menghapus stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

dataTweet['Tweet_tokens_WSW'] = dataTweet['Tweet_tokens'].apply(stopwords_removal)
# print(dataTweet['Tweet_tokens_WSW'].head(10))


#--- Normalization ---#
normalizad_word = pd.read_excel("Book1.xlsx")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]

def normlized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

dataTweet['Tweet_normalized'] = dataTweet['Tweet_tokens_WSW'].apply(normlized_term)
dataTweet['Tweet_normalized'].head(10)

# Membuat Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in dataTweet['Tweet_normalized']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '


print(len(term_dict))
print("--------------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":", term_dict[term])

print(term_dict)
print("--------------------------------")

# Apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

dataTweet['Tweet_tokens_stemmed'] = dataTweet['Tweet_normalized'].swifter.apply(get_stemmed_term)
print(dataTweet['Tweet_tokens_stemmed'].head(10))


