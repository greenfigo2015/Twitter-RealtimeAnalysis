from pubnub.callbacks import SubscribeCallback
from pubnub.enums import PNOperationType, PNStatusCategory, PNReconnectionPolicy
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub import PubNub, SubscribeListener
from pubnub.exceptions import PubNubException
import sys
import nltk
import re
import pandas as pd
import os
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from textstat.textstat import *
import json



pn_twitter_config = PNConfiguration()
pn_twitter_config.subscribe_key = 'sub-c-78806dd4-42a6-11e4-aed8-02ee2ddab7fe'
pn_twitter_config.reconnect_policy = PNReconnectionPolicy.EXPONENTIAL
pubnub_twitter = PubNub(pn_twitter_config)

pn_config = PNConfiguration()
pn_config.subscribe_key = 'sub-c-93679206-1202-11e8-bb84-266dd58d78d1'
pn_config.publish_key = 'pub-c-44842793-32df-4b69-b576-4495abcf64ec'
pubnub_personal = PubNub(pn_config)



#TF hack
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Change Folder Path here 
raw_data_path = "/Users/ritiztambi/repos/Twitter-RealtimeAnalysis-Pubnub"  
os.chdir(raw_data_path)
    
#Creating dataframes
df = pd.read_csv('data/labeled_data.csv',error_bad_lines=False)
df1= pd.read_csv('data/labeled_data_validate.csv',error_bad_lines=False)
df2= pd.read_csv('data/labeled_test.csv',error_bad_lines=False)
tweets_train=df.tweet
tweets_validate=df1.tweet
tweets_test=df2.tweet


#stopwords for training
stopwords=stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)


stemmer = PorterStemmer()
sentiment_analyzer = VS()


def preprocess(tweet):
    """Removes Url,non alphanumeric symbols,hashtags,emojis (For feature transform)"""   
    #Remove URL
    tweet = re.sub('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)   
    #Remove user
    tweet = re.sub('@[^\s]+','',tweet)
    #Remove not alphanumeric symbols white spaces
    tweet = re.sub(r'[^\w]', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', ' ', tweet)                 
    tweet = re.sub('[:;>?<=*+()/,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]',' ', tweet)
    #Remove :( or :)
    tweet = tweet.replace(':)','')
    tweet = tweet.replace(':(','')     
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('[\n]+', ' ', tweet)
    return tweet
    

def clean_tweet(tweet):
    """Removes Url,non alphanumeric symbols,hashtags,emojis (For sentiment analysis)"""   
    #Remove URL
    tweet = re.sub('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', tweet)   
    #Remove user
    tweet = re.sub('@[^\s]+','',tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', ' ', tweet)              
    return tweet


def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'),parsed_text.count('MENTIONHERE'),parsed_text.count('HASHTAGHERE'))


def tokenize(tweet):
    """This function removes punctuation & excess whitespace, sets to 
       lowercase, stems tweets and teturns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens



""" Vectorize tweets."""
vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    #min_df=5,
    #max_df=0.75
    )


# extra features borrowed from the paper https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665
def sent_features(tweet):
    """This function takes a tweet as a string and returns a list of 
       Sentiment scores as features"""
    twitter_objs = count_twitter_objs(tweet)
    tweet=clean_tweet(tweet)   
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    #Get text only
    words = preprocess(tweet) 
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    
    \
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE,syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    return features
  

def get_feature_array(tweets):
    """This function takes a string of tweets and returns a list of 
       Sentiment scores as features"""
    feats=[]
    for t in tweets:
        feats.append(sent_features(t))
    return np.array(feats)


def get_features_train(tweets):
    """This function takes a string of tweets and returns a featureSet. 
       (For Training data)"""
    feats = get_feature_array(tweets)
    tfidf = vectorizer.fit_transform(tweets).toarray()
    M = np.concatenate([tfidf,feats],axis=1)
    return M

def get_features_test(tweets):
    """This function takes a string of tweets and returns a featureSet. 
       (For Test data)"""
    feats = get_feature_array(tweets)
    tfidf = vectorizer.transform(tweets).toarray()
    M = np.concatenate([tfidf,feats],axis=1)
    return M


    


""" Get Train featureSet, Test featureSet, Train topic class and Test topic class  """    
X_train = pd.DataFrame(get_features_train(tweets_train))
y_train = df['class'].astype(int)
X_validate =  pd.DataFrame(get_features_test(tweets_validate))
y_validate =  df1['class'].astype(int)
X_test =  pd.DataFrame(get_features_test(tweets_test))
y_test =  df2['class'].astype(int)


number_of_classes=3
input_dimention = X_train.shape[1]

#Convert class vector to binary class matrix.
onehot_train = np_utils.to_categorical(y_train, number_of_classes)  
onehot_validate = np_utils.to_categorical(y_validate, number_of_classes)  
onehot_test =   np_utils.to_categorical(y_test, number_of_classes)  

print('tweet_train shape:', X_train.shape)
print('tweet_validate shape:', X_validate.shape)
print('tweet_test shape:', onehot_train.shape)
print('tweet_train_class shape:', onehot_train.shape)
print('tweet_validate_class shape:', onehot_validate.shape)
print('tweet_test_class shape:', onehot_test.shape)


""" Define model """
model = Sequential()
model.add(Dense(256, input_dim=input_dimention,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(number_of_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("Training model")
model.fit(X_train, onehot_train, epochs=1, batch_size=32,validation_data=(X_validate,onehot_validate))
print("Generating test Predictions, Accuracy")
predited_topics = model.predict_classes(X_test)
accuracy_of_test = accuracy_score(y_test, predited_topics)
print ('Accuracy is ',accuracy_of_test*100,'%')
    
def predict_topic(tweet):
    """This function takes a tweet and returns the predicted class as . 
       0-hate speech, 1-offensive language, 2-neither"""
    topic =  model.predict_classes(pd.DataFrame(get_features_test([tweet])))
    if topic == 0:
        return 'hate_speech'
    elif topic ==1:
        return 'offensive'
    elif topic ==2:
        return 'neither'
 
    	
        
def get_lang(tweet):
    """This function takes a tweet and tries to determine its language""" 
    lang_ratios = {}
    #tokenize tweet
    token = word_tokenize(tweet)
    words = [word.lower() for word in token]
    #match tweet stopwords for each language
    for lang in nltk.corpus.stopwords.fileids():
        stopwords_set = set(nltk.corpus.stopwords.words(lang))
        words_set=set(words)
        ele_common=words_set.intersection(stopwords_set)
        lang_ratios[lang] = len(ele_common)
            
    return max(lang_ratios,key=lang_ratios.get)
        


def get_tweet_sentiment(tweet):
    """This function takes a tweet and tries to determine its language"""
    #Process Tweet
    tweet=clean_tweet(tweet)  
    #get sentiment
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    analysis = sentiment['compound']
    # return sentiment
    if analysis >= 0.5:
        return 'positive'
    elif analysis <= -0.5:
        return 'negative'
    else:
        return 'neutral'


def my_publish_callback(envelope, status):
    # Check whether request successfully completed or not
    if not status.is_error():
        pass  # Message successfully published to specified channel.
    else:
        pass  



class TwitterSubscribeCallback(SubscribeCallback):

   
    count = 0
    
    #Initializing bucket
    bucket = {}
    bucket['neither']=0
    bucket['offensive']=0
    bucket['hate_speech']=0
    
    total_sentiment = 0
    
    #Initializing sentiment bucket
    bucket_sentiment={}
    bucket_sentiment['negative']=0
    bucket_sentiment['positive']=0
    bucket_sentiment['neutral']=0
    
    Eon={}

  
    def populate_bucket(self,topic):
        """This function populates the text-analysis bucket with each 
           incoming tweet"""
        if topic == 'neither':
            self.bucket['neither']+=1
        elif topic == 'offensive':
            self.bucket['offensive']+=1
        else:
            self.bucket['hate_speech']+=1
        
    def populate_sentiment(self,sentiment):
        """This function populates the sentiment-analysis bucket with each 
           incoming tweet"""
        if sentiment == 'neutral':
            self.bucket_sentiment['neutral']+=1
        elif sentiment == 'negative':
            self.bucket_sentiment['negative']+=1
        else:
            self.bucket_sentiment['positive']+=1
    
    
    def presence(self, pubnub, presence):
        pass  # handle incoming presence data
 
    
    def status(self, pubnub, status):
        if status.category == PNStatusCategory.PNUnexpectedDisconnectCategory:
             # This event happens when radio / connectivity is lost 
            print("Retrying")
            pubnub.reconnect()
           
        elif status.category == PNStatusCategory.PNConnectedCategory:
            # Connect event.
            print('Hello')
            pass
        elif status.category == PNStatusCategory.PNReconnectedCategory:
            # This event happens when radio / connectivity is lost
            print("Reconnected")
            pass
            


    # Where we filter tweets and publish for sentiment analysis
    def message(self, pubnub, message):
            tweet=message.message['text']
            #get tweet language and process further only if 
            language=get_lang(tweet)
            if(language == 'english'):
                cls=predict_topic(tweet)
                self.populate_bucket(cls)
                tweet=clean_tweet(tweet)
                sentiment=get_tweet_sentiment(tweet)
                self.total_sentiment+=1;
                self.populate_sentiment(sentiment) 
                try:

                    pubnub_personal.publish().channel("sentiment_channel").message({
                        'positive': self.bucket_sentiment['positive']/self.total_sentiment,
                        'negative': self.bucket_sentiment['negative']/self.total_sentiment,
                        'neutral' : self.bucket_sentiment['neutral']/self.total_sentiment
                    }).async(my_publish_callback)
    
                    pubnub_personal.publish().channel("topic_channel").message({
                        'neither': self.bucket['neither'],
                        'offensive': self.bucket['offensive'],
                        'hate_speech' : self.bucket['hate_speech']
                    }).async(my_publish_callback)
                   
                except PubNubException as e:
                    print(e)
            else:
                pass
            self.count=self.count+1
    

          #  pubnubPersonal.publish().channel("sentiment-analysis").message({"session_id":self.session_Id,"text":message.message['text']}).async(my_publish_callback)
       
        

class MySubscribeCallback(SubscribeCallback):

    
    def presence(self, pubnub, presence):
        pass  # handle incoming presence data
 
    
    def status(self, pubnub, status):
        if status.category == PNStatusCategory.PNUnexpectedDisconnectCategory:
             # This event happens when radio / connectivity is lost 
            print("Retrying")
            pubnub.reconnect()
           
        elif status.category == PNStatusCategory.PNConnectedCategory:
            # Connect event.
            print('Hello Personal')
            pass
        elif status.category == PNStatusCategory.PNReconnectedCategory:
            # This event happens when radio / connectivity is lost
            print("Reconnected")
            pass
            


    # Where we filter tweets and publish for sentiment analysis
    def message(self, pubnub, message):        
            if message.message == 'start':
                pubnub_twitter.subscribe().channels('pubnub-twitter').execute()
            elif message.message == 'stop':
                pubnub_twitter.unsubscribe().channels('pubnub-twitter').execute()  
        

        


pubnub_twitter.add_listener(TwitterSubscribeCallback())
pubnub_personal.add_listener(MySubscribeCallback())
pubnub_personal.subscribe().channels('start-stop').execute()






    

 

 

