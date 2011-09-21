from collections import defaultdict
from operator import itemgetter

class NaiveBayesClassifier(object):
    def __init__(self):
        self.label_totals = defaultdict(int)
        self.feature_totals = defaultdict(lambda: defaultdict(int))
        self.num_training_instances = 0
    
    @classmethod
    def features(cls, instance, label=None):
        raise NotImplementedError('Children must implement their own')
    
    def expected_likelihood_estimate(self, feature, label):
        """ For unseen features (P(feature|class)=0), adds a constant of
        0.5 / 0.5*num_classes to make it a non-zero probability.
        Similar to the Laplace estimator but valuing unseen values a little less"""
        
        # Doing this to avoid creating extra keys in defaultdict from unseen instances
        feature_given_label = self.feature_totals.get(feature, {}).get(label, 0)
        
        return (0.5 + feature_given_label ) / (0.5* len(self.label_totals.keys()) + self.label_totals[label] )
    
    def prob_classify(self, inst):
        """Returns a dict like {label: probability} where probabilities sum to 1"""
        prob_dist = dict.fromkeys(self.label_totals.keys(), 1.0)
        features = self.features(inst)

        for label in self.label_totals:        
            for feature, value in features.iteritems():
                # P(label|feature)
                prob_dist[label] *= self.expected_likelihood_estimate(feature, label)
            
            # P(label)
            prob_dist[label] *= (float(self.label_totals[label]) / sum(self.label_totals.values()))

        normalizer = sum(prob_dist.values())
        
        for label in prob_dist:
            prob_dist[label] = prob_dist[label] / normalizer

        return prob_dist                
    
    def classify(self, inst):
        """ Max of the probability distribution """
        return sorted(self.prob_classify(inst).iteritems(),
                      key=lambda item: item[1], reverse=True)[0][0]
    
    def train(self, data):
        """Since we're only storing counts, this can be updated iteratively"""
        for inst, label in data:
            self.label_totals[label] += 1
            self.num_training_instances += 1
            
            features = self.features(inst, label)
            for feature, value in features.iteritems():
                self.feature_totals[feature][label] += value
               
    def accuracy(self, gold):
        correct = total = 0
        for inst, label in gold:
            predicted = self.classify(inst)
            if predicted == label:
                correct += 1
                
            total += 1
        
        return float(correct) / total
        
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
    data = [('I hate Obama, taxes are bad for America', 'GOP'),
        ('Americans love @BarackObama', 'DEM')]
    
    class MyClassifier(NaiveBayesClassifier):
        @classmethod
        def features(cls, inst, label=None):
            return dict((word, True) for word in inst.split())
    
    model = MyClassifier()
    model.train(data)
    print model.prob_classify('I hate Barack Obama')
    print model.classify('I hate Barack Obama')
