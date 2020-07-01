# imports
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from datetime import datetime
import json
import tweepy as tw

# consumer key
consumer_key = ""
# consumer secrety
consumer_secrety = ""
# access token
access_token = ""
# access token secrety
access_token_secrety = ""


# auth
auth = OAuthHandler(consumer_key, consumer_secrety)
auth.set_access_token(access_token, access_token_secrety)

