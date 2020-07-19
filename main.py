# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import brown
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
from nltk.stem.wordnet import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.text import text_to_word_sequence
import string                              # for string operations
import re
import copy
import math
import os
import sys
import traceback
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', 
                    filename='log.txt', filemode='w', level=logging.DEBUG, 
                    datefmt='%Y-%m-%d %H:%M:%S')

def process_tweet(tweet, stopwords, punctuations):
    """
    process twitter text by removing retweet signature(RT), url, hashtags, stop words, punctuations

    Parameters
    ----------
    tweet : str
        Text content of a tweet

    Returns
    -------
    tweet_token_processed : list of str
        List of tokens of the tweet after it is processed 

    """
    retweetRT_removed = re.sub(r'^RT[\s]','',tweet)
    tweet_url_removed = re.sub(r'https?:\/\/.*[\r\n]*','',retweetRT_removed)
    tweet_hashtag_removed = re.sub(r'#', '', tweet_url_removed)    
    
    # instantiate tweettokenizer class
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    # tokenize the tweet
    tweet_tokens = tokenizer.tokenize(tweet_hashtag_removed)    
    
    #remove stop word and punctuations
    tweet_token_processed = [item for item in tweet_tokens if ((item not in stopwords) and (item not in punctuations))]

    #lemmatizing words using wordnet lemmatizer    
    lmtzr = WordNetLemmatizer()
    tweet_tokens_lmtzd = [lmtzr.lemmatize(item) for item in tweet_token_processed]
    
    return tweet_tokens_lmtzd


def create_freq_dict_and_unique_word_list(positive_tweets_processed, negative_tweets_processed):
    #create (word,label):frequency dictionary
    word_label_dict = {}
    unique_words = set()

    for item in [item for sublist in positive_tweets_processed for item in sublist]:
        unique_words.add(item)
        if (item,1) in word_label_dict:
            word_label_dict[(item,1)] +=1
        else:
            word_label_dict[(item,1)] = 1

    for item in [item for sublist in negative_tweets_processed for item in sublist]:
        unique_words.add(item)
        if (item,0) in word_label_dict:
            word_label_dict[(item,0)] +=1
        else:
            word_label_dict[(item,0)] = 1

    return [word_label_dict, unique_words]


def create_word_vec_dict(unique_words, word_label_dict):
    word_vec_dict={}
    for item in unique_words:
        try:
            pos_count = word_label_dict[(item,1)]
        except:
            pos_count = 0
           
        try:   
            neg_count = word_label_dict[(item,0)]
        except:
            neg_count = 0

        word_vec_dict[item] = [pos_count, neg_count]
    return word_vec_dict

    

if __name__ == "__main__":
    #download twitter_sample data, and list of stopwords and punctuations for pre-processing the text
    # nltk.download("twitter_samples")
    # nltk.download("wordnet")
    stopwords = stopwords.words('english')
    punctuations = string.punctuation
    
    #Load the data to variables using "strings" method
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    
    positive_tweets_processed = [process_tweet(item, stopwords, punctuations) for item in positive_tweets]
    negative_tweets_processed = [process_tweet(item, stopwords, punctuations) for item in negative_tweets]            

    #create (word,label):frequency dictionary, and return unique word vector
    [word_label_dict, unique_words] = create_freq_dict_and_unique_word_list(positive_tweets_processed, negative_tweets_processed)
        
    #Construct word vectors
    word_vec_dict = create_word_vec_dict(unique_words, word_label_dict)
    
        
    
    