#!/usr/bin/env python
# encoding: utf-8
import datetime

import tweepy
import csv

# Twitter API credentials
consumer_key = "QlvO5PLHOpjBnOoTkNS7IWIdV"
consumer_secret = "61GLkCOQw3RPuEVChFrFubSltOLerJLet5hGZvEmFJJGPVOpH2"
access_key = "4498784320-ALzJ8W8Aps2aja8eej7Iu9pK0WuaCieFjf7InFV"
access_secret = "MF7rur9RzXwc1f2L8rzTavDcCkVW3p4agsqWWOBWoUjVQ"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)


def get_all_tweets(screen_name):
    tweets = ''
    username = screen_name
    count = 500
    startDate = datetime.datetime(2020, 6, 1, 0, 0, 0)
    endDate = datetime.datetime(2015, 1, 1, 0, 0, 0)
    for tweet in api.user_timeline(id=username, count=count):
        tweets += tweet.text

    return tweets


if __name__ == '__main__':
    # pass in the username of the account you want to download
    #get_all_tweets("realDonaldTrump")
    get_all_tweets('realDonaldTrump')