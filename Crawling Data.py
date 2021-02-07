import tweepy
import pandas as pd

# Token, Token Secret, Key, dan Key Secret
customer_key = 'FNqbHb2nUIeTVvlBRysDsCo5e'
customer_Secret = 'yVLlSAzPTN5BMwyte1JHZ4GpqAYqVIVIrKweiLAV0oEcL0PQ98'
access_token = '741837277844758528-PJeAHdh0jVBKvDtL5QmSwRdFmBf969l'
access_token_secret = '2KuVZikDNzNpcSvYEtVtdRzmhH4bM97X5RMCmA36DbKHV'

# Konfigurasi Pencarian
pencarian_hashtag = "#dirumahaja"
keyword = pencarian_hashtag + " -filter:retweets "
jumlah_tweet = 100
# data_since = "2020-05-20"

auth = tweepy.OAuthHandler(customer_key, customer_Secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

tweets = tweepy.Cursor(api.search,
                       q= keyword,
                       full_text= True,
                       result_type= "recent",
                       tweet_mode= 'extended',
                       lang= "id").items(jumlah_tweet)


# Looping data Twitter
# datatweeet = [[tweet.user.name,tweet.user.screen_name,tweet.full_text] for tweet in tweets]
datatweeet = [tweet.full_text for tweet in tweets]

# Membuat dataframwe dengan Pandas
# df = pd.DataFrame(data=datatweeet,
#                   columns=['Name', 'Screen Name', 'Tweet'])
df = pd.DataFrame(data=datatweeet,
                  columns=['Tweet'])

# Export data Twitter dengan format csv menggunakan Pandas
df.to_csv("data_twitter.csv")
df.to_csv("data_twitter2.csv", sep='\t', encoding='utf-8')


