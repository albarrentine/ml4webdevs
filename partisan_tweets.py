#!/usr/bin/env python

import tweepy
import nltk
import os
from operator import itemgetter
from itertools import chain
from naive_bayes import NaiveBayesClassifier
from functools import partial

from nltk.stem.porter import PorterStemmer

hashtags = {'DEM': '#p2',
            'GOP': '#tcot'}

# Use Boolean queries to give us set complements (exclude the intersection
# or tweets that are tagged both #tcot and #p2), at least for the training data

queries = {'DEM': '%s -%s' % (hashtags['DEM'], hashtags['GOP']),
           'GOP': '%s -%s' % (hashtags['GOP'], hashtags['DEM'])
           }

api = tweepy.API()

stopwords = set([word.rstrip() for word in open('stopwords.txt')]) | set(['.',',',';','"',"'",'?','(',')',':','-','_','`','...'])
porter = PorterStemmer()


class PartisanTweetClassifier(NaiveBayesClassifier):
    def __init__(self, filename='twitter_model.out', **kw):
        super(PartisanTweetClassifier, self).__init__(**kw)
        self.filename = filename
        self.training_sets = {}
        self.test_sets = {}
    
    @classmethod
    def features(cls, tweet, label=None):
        """ Fill in your own method. Here's an example to get you started """
        
        text = tweet if isinstance(tweet, basestring) else tweet.text
        
        hashtag = hashtags.get(label)
        
        words = [porter.stem(word.lower()) if not word.startswith('#') or word.startswith('@') else word.lower() for word in text.split()
                 if word.lower() != hashtag and word not in stopwords]
        
        f = dict((word, True) for word in words)
        f.update(dict((' '.join(word), True) for word in nltk.bigrams(words)))
        #f.update(dict((' '.join(word),True) for word in nltk.trigrams(words)))
        
        return f
        
    def fetch_twitter(self):
        for party, hashtag in hashtags.iteritems():
            print 'Doing party %s' % party
            print 'Fetching first page'
            training_tweets = api.search(q=queries[party], rpp=200)
            max_id=training_tweets[0].id
            
            res = [(tweet, party) for tweet in training_tweets]
            
            for page in range(2,8):
                print 'Fetching page %d' % page
                res.extend([(tweet, party) for tweet in api.search(q=queries[party], rpp=200, page=page, max_id=max_id)])  

            # Delete the hashtag itself since it is effectively the label in this case
            for tweet, party in res:
                tweet.text = tweet.text.replace(hashtags[party], '')
            
            middle = len(res)/2
            self.training_sets[party] = res[:middle]
            self.test_sets[party] = res[middle+1:]
            
    def salient_features(self, n=100, pprint=False):
        ratios = {}
        labels = self.label_totals.keys()
        
        for feature, prob_dict in self.feature_totals.iteritems():
            probs = [self.expected_likelihood_estimate(feature, label) for label in labels]
            min_prob = min(enumerate(probs), key=itemgetter(1))
            max_prob = max(enumerate(probs), key=itemgetter(1))
            ratios[feature] = (labels[max_prob[0]], int(round(max_prob[1] / min_prob[1])) )

        probs = sorted(ratios.iteritems(), key=lambda item: item[1][1], reverse=True)[:n]
        
        if pprint:
            print 'Most salient features:'
            for feature, (label, ratio) in probs:
                print ('%24s %6s = %s : 1.0') % (feature, label, ratio)
        else:
            return probs
           
if __name__ == '__main__':
    from itertools import chain
    import time
    c = PartisanTweetClassifier()
    print 'Fetching Twitter data...'
    c.fetch_twitter()
    print 'Training model'
    c.train(chain(*c.training_sets.values()))
    c.salient_features(pprint=True)
    print 'Accuracy: %s' % c.accuracy(chain(*c.test_sets.values()))
