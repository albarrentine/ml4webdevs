from flask import Flask, render_template, render_template_string, jsonify, request
from friend_similarity import FriendSimilarity
from partisan_tweets import PartisanTweetClassifier
from itertools import chain

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/partisan_tweets', methods=['GET'])
def partisan_tweets():
    return render_template('partisan_tweets.html')

@app.route('/partisan_tweets', methods=['POST'])
def post_partisan_tweets():     # LOL method name
    c = PartisanTweetClassifier()
    c.fetch_twitter()
    c.train(chain(*c.training_sets.values()))
    accuracy = c.accuracy(chain(*c.test_sets.values()))
    salient = c.salient_features()
    
    html = render_template('tweets.html', accuracy=accuracy, salient = salient)
    return jsonify({'result': html})
    
@app.route('/friend_similarity', methods=['GET'])
def friend_similarity():
    return render_template('friend_similarity.html')

@app.route('/friend_similarity', methods=['POST'])
def post_friend_similarity():
    model = FriendSimilarity(cookies = request.cookies)
    me, friends = model.most_similar()
    html = render_template('friends.html', me=me, friends=friends)
    return jsonify({'result': html})


if __name__ == '__main__':
    app.debug = True
    app.run()
