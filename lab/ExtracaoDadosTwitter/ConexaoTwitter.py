# imports
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from datetime import datetime
import json
import tweepy as tw

# consumer key
consumer_key = "yEx7mCjjgM9Z9hez4ZP1kL5xV"
# consumer secrety
consumer_secrety = "cno7DN2EHvyc7ZKtg6a91yRnAh6oqzvYTiRF2my4SKvRgoUdlr"
# access token
access_token = "1267792852551118851-SbFDhj0Do77ZCzIv8TMm5hG158BJ57"
# access token secrety
access_token_secrety = "kSp2fVXSMS9N49byhaetSGydCF6T35uGxI949ob39gz0f"


# auth
auth = OAuthHandler(consumer_key, consumer_secrety)
auth.set_access_token(access_token, access_token_secrety)

