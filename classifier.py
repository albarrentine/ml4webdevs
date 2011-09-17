#!/usr/bin/env python

import tweepy
import nltk
import yaml
from itertools import chain

from nltk.stem.porter import PorterStemmer

hashtags = {'DEM': '#p2',
            'GOP': '#tcot'}

# Use Boolean queries to give us set complements (exclude the intersection
# or tweets that are tagged both #tcot and #p2), at least for the training data

queries = {'DEM': '%s -%s' % (hashtags['DEM'], hashtags['GOP']),
           'GOP': '%s -%s' % (hashtags['GOP'], hashtags['DEM'])
           }

api = tweepy.API()

def features(tweet, hashtag):
    """ Fill in your own method. Here's an example to get you started """
    return dict((word.lower(), True) for word in tweet.text.split() if word != hashtag)
    
class PartisanTweetClassifier(object):
    def __init__(self, output_file=None):
        self.output_file = output_file
        self.model = None
        self.training_sets = {}
        self.test_sets = {}
        
    def fetch_twitter(self):
        # Get 200 tweets for each hashtag to train and the next 200 for testing the model's accuracy  
        for party, hashtag in hashtags.iteritems():
            print 'Doing party %s' % party
            print 'Fetching first page'
            training_tweets = api.search(q=queries[party], rpp=200)
            max_id=training_tweets[0].id
            
            res = [(features(tweet, hashtag), party) for tweet in training_tweets]
            
            for page in range(2,8):
                print 'Fetching page %d' % page
                res.extend([(features(tweet, hashtag), party) for tweet in api.search(q=queries[party], rpp=200, page=page, max_id=max_id)])  
 
            middle = len(res)/2
            self.training_sets[party] = res[:middle]
            self.test_sets[party] = res[middle+1:]
        
    def train_model(self):
        self.model = nltk.classify.NaiveBayesClassifier.train(list(chain(*self.training_sets.values())))
        
    def classify_tweet(self, tweet):
        return self.model.classify(features(tweet, None))
        
    def classify_text(self, text):
        return self.model.classify(dict((word.lower(), True) for word in text.split()))

    def save(self):
        if self.output_file is not None:
            self.output_file.write(yaml.dump(self.model))
    
    def accuracy(self):
        return nltk.classify.accuracy(self.model, list(chain(*self.test_sets.values())))
    
    def salient_features(self):
        self.model.show_most_informative_features(n=50)