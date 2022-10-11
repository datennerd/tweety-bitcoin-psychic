"""Trainings pipeline: Handle the Twitter API with tweepy."""

import os
from datetime import datetime, timedelta

import config
import tweepy

# Twitter API Secrets
API_KEY = os.environ["API_KEY"]
API_SECRET_KEY = os.environ["API_SECRET_KEY"]
ACCESS_TOKEN = os.environ["ACCESS_TOKEN"]
ACCESS_TOKEN_SECRET = os.environ["ACCESS_TOKEN_SECRET"]

# Authenticate to Twitter
auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Create API object
api = tweepy.API(auth)

# Create the message
today = datetime.now().strftime("%Y-%m-%d")
sevendays = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
message = f"Daily #Bitcoin-USD closing price forecast: {today} to {sevendays}"

# Post a tweet
tweet = api.update_status_with_media(message, f"{config.OUTPUT_PATH}/forecast.png")
