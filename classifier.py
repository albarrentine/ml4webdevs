#!/usr/bin/env python

import tweepy
import nltk
import yaml
from itertools import chain

from nltk.classify.api import ClassifierI
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
        
        
# Copy of the implementation
class NaiveBayesClassifier(ClassifierI):
    """
    A Naive Bayes classifier.  Naive Bayes classifiers are
    paramaterized by two probability distributions:

      - P(label) gives the probability that an input will receive each
        label, given no information about the input's features.
        
      - P(fname=fval|label) gives the probability that a given feature
        (fname) will receive a given value (fval), given that the
        label (label).

    If the classifier encounters an input with a feature that has
    never been seen with any label, then rather than assigning a
    probability of 0 to all labels, it will ignore that feature.

    The feature value 'None' is reserved for unseen feature values;
    you generally should not use 'None' as a feature value for one of
    your own features.
    """
    def __init__(self, label_probdist, feature_probdist):
        """
        @param label_probdist: P(label), the probability distribution
            over labels.  It is expressed as a L{ProbDistI} whose
            samples are labels.  I.e., P(label) =
            C{label_probdist.prob(label)}.
        
        @param feature_probdist: P(fname=fval|label), the probability
            distribution for feature values, given labels.  It is
            expressed as a dictionary whose keys are C{(label,fname)}
            pairs and whose values are L{ProbDistI}s over feature
            values.  I.e., P(fname=fval|label) =
            C{feature_probdist[label,fname].prob(fval)}.  If a given
            C{(label,fname)} is not a key in C{feature_probdist}, then
            it is assumed that the corresponding P(fname=fval|label)
            is 0 for all values of C{fval}.
        """
        self._label_probdist = label_probdist
        self._feature_probdist = feature_probdist
        self._labels = label_probdist.samples()

    def labels(self):
        return self._labels

    def classify(self, featureset):
        return self.prob_classify(featureset).max()
        
    def prob_classify(self, featureset):
        # Discard any feature names that we've never seen before.
        # Otherwise, we'll just assign a probability of 0 to
        # everything.
        featureset = featureset.copy()
        for fname in featureset.keys():
            for label in self._labels:
                if (label, fname) in self._feature_probdist:
                    break
            else:
                #print 'Ignoring unseen feature %s' % fname
                del featureset[fname]

        # Find the log probabilty of each label, given the features.
        # Start with the log probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_probdist.logprob(label)
            
        # Then add in the log probability of features given labels.
        for label in self._labels:
            for (fname, fval) in featureset.items():
                if (label, fname) in self._feature_probdist:
                    feature_probs = self._feature_probdist[label,fname]
                    logprob[label] += feature_probs.logprob(fval)
                else:
                    # nb: This case will never come up if the
                    # classifier was created by
                    # NaiveBayesClassifier.train().
                    logprob[label] += sum_logs([]) # = -INF.
                    
        return DictionaryProbDist(logprob, normalize=True, log=True)

    def show_most_informative_features(self, n=10):
        # Determine the most relevant features, and display them.
        cpdist = self._feature_probdist
        print 'Most Informative Features'

        for (fname, fval) in self.most_informative_features(n):
            def labelprob(l):
                return cpdist[l,fname].prob(fval)
            labels = sorted([l for l in self._labels
                             if fval in cpdist[l,fname].samples()],
                            key=labelprob)
            if len(labels) == 1: continue
            l0 = labels[0]
            l1 = labels[-1]
            if cpdist[l0,fname].prob(fval) == 0:
                ratio = 'INF'
            else:
                ratio = '%8.1f' % (cpdist[l1,fname].prob(fval) /
                                  cpdist[l0,fname].prob(fval))
            print ('%24s = %-14r %6s : %-6s = %s : 1.0' %
                   (fname, fval, str(l1)[:6], str(l0)[:6], ratio))

    def most_informative_features(self, n=100):
        """
        Return a list of the 'most informative' features used by this
        classifier.  For the purpose of this function, the
        informativeness of a feature C{(fname,fval)} is equal to the
        highest value of P(fname=fval|label), for any label, divided by
        the lowest value of P(fname=fval|label), for any label::

          max[ P(fname=fval|label1) / P(fname=fval|label2) ]
        """
        # The set of (fname, fval) pairs used by this classifier.
        features = set()
        # The max & min probability associated w/ each (fname, fval)
        # pair.  Maps (fname,fval) -> float.
        maxprob = defaultdict(lambda: 0.0)
        minprob = defaultdict(lambda: 1.0)

        for (label, fname), probdist in self._feature_probdist.items():
            for fval in probdist.samples():
                feature = (fname, fval)
                features.add( feature )
                p = probdist.prob(fval)
                maxprob[feature] = max(p, maxprob[feature])
                minprob[feature] = min(p, minprob[feature])
                if minprob[feature] == 0:
                    features.discard(feature)

        # Convert features to a list, & sort it by how informative
        # features are.
        features = sorted(features, 
            key=lambda feature: minprob[feature]/maxprob[feature])
        return features[:n]

    @staticmethod
    def train(labeled_featuresets, estimator=ELEProbDist):
        """
        @param labeled_featuresets: A list of classified featuresets,
            i.e., a list of tuples C{(featureset, label)}.
        """
        label_freqdist = FreqDist()
        feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()

        # Count up how many times each feature value occured, given
        # the label and featurename.
        for featureset, label in labeled_featuresets:
            label_freqdist.inc(label)
            for fname, fval in featureset.items():
                # Increment freq(fval|label, fname)
                feature_freqdist[label, fname].inc(fval)
                # Record that fname can take the value fval.
                feature_values[fname].add(fval)
                # Keep a list of all feature names.
                fnames.add(fname)

        # If a feature didn't have a value given for an instance, then
        # we assume that it gets the implicit value 'None.'  This loop
        # counts up the number of 'missing' feature values for each
        # (label,fname) pair, and increments the count of the fval
        # 'None' by that amount.
        for label in label_freqdist:
            num_samples = label_freqdist[label]
            for fname in fnames:
                count = feature_freqdist[label, fname].N()
                feature_freqdist[label, fname].inc(None, num_samples-count)
                feature_values[fname].add(None)

        # Create the P(label) distribution
        label_probdist = estimator(label_freqdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label,fname] = probdist

        return NaiveBayesClassifier(label_probdist, feature_probdist)
