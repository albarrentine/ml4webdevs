import urllib, urllib2
import facebook
from collections import defaultdict
from nltk.stem.porter import PorterStemmer

APP_ID = '207747259291281'
APP_SECRET = '8fc4c463734694f163e2a249ef6edacd'

porter = PorterStemmer()

class FriendSimilarity(object):
    def __init__(self, cookies=None, access_token=None):
        if cookies:
            user = facebook.get_user_from_cookie(cookies, APP_ID, APP_SECRET)
            access_token = user['access_token']
        
        self.api = facebook.GraphAPI(access_token)

    def jaccard_coefficient(self, set1, set2):
        """ Using the naive method. On large datasets you typically want to use minhashing """
        return float(len(set1.intersection(set2))) / len(set1.union(set2))
        
    def most_similar(self, n=25):
        me, my_interests = self.get_me_and_interests()
        friends, friends_interests = self.get_friends_and_interests()
        
        jaccard = {}
        for id, their_interests in friends_interests.iteritems():
            jaccard[id] = self.jaccard_coefficient(my_interests, their_interests)
        
        top_n_friends = sorted(jaccard.iteritems(), key = lambda item: item[1], reverse=True)[:n]
        ret = [dict(id=id, name=friends[id]['name'], picture=friends[id]['picture'],
                    interests=friends_interests[id], sim=similarity) for id, similarity in top_n_friends]
        """
        If you're interested, you can also do this:
        
        friend_sims = {}
        friend_ids = friends_interests.keys()
        
        for idx, friend1 in enumerate(friend_ids):
            for friend2 in friend_ids[idx+1:]:
                friend_sims[(friend1, friend2)] = self.jaccard_coefficient(friends_interests[friend1], friends_interests[friend2])
                        
        most_similar_friends = sorted(friend_sims.iteritems(), key = lambda item: item[1], reverse=True)[:n]
        friend_ret = [(friends[id1]['name'], friends[id2]['name'], sim) for (id1, id2), sim in most_similar_friends]
        
        return ret, friend_ret
        
        """
        
        return me, ret

    @classmethod
    def parse_people_and_interests(cls, data):
        all_interests = defaultdict(set)
        people_dict = {}
        
        if not data:
            return {}
            
        for person in data:            
            id = person.get('id')
            people_dict[id] = {'name': person['name'], 'picture': person['picture']}

            interests = person.get('interests')
            
            if interests:
                interests = interests['data']
                for interest in interests:
                    """ Do anything you want here, I'm just lower-casing and stemming.
                    You can also add additional data to the set to get better results"""
                    all_interests[id].add((interest['category'], tuple([porter.stem(word.lower()) for word in interest['name'].split()] ) ) )
        
        return people_dict, all_interests

    def get_me_and_interests(self):
        data = self.api.get_object('me', fields='name, picture, interests')
        my_id = data['id']
        ret = self.parse_people_and_interests([data])
        return ret[0][my_id], ret[1][my_id]
    
    def get_friends_and_interests(self):
        data = self.api.get_connections('me', 'friends', fields='name, picture, interests').get('data', None)
        return self.parse_people_and_interests(data)

