

```python
# News Mood: 
#     Social media has broad-sweeping effects on society. In this project we will perform a sentiment analysis for 
#     100 tweets for five major news/media outlets (BBC, CNN, CBS, FoxNews, NY Times) at any given time in order 
#     to determine whether they are setting a positive or negative tone for the day. 
    
#     Sentiment scores range from 1 (Positive), and -1 (Negative). While this is useful for determining the general 
#     tone of a singular tweet in a series of tweets, the average (Compound) score serves as a better indicator 
#     of the overall tone of a series. 
    
#     The purpose of this exercise is to find out - simply - what is the News' Mood today? 
    
```


```python
# Dependencies 

import os 
import pandas as pd 
import tweepy 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import time
import requests as req
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
analyzer = SentimentIntensityAnalyzer()
```


```python
# API keys & Tweepy Authentication 

# Twitter API Keys
consumer_key = "uqWxn83alfwZNumLlvhO2Wv3c"
consumer_secret = "bttQrkHniBnVK1Ejiz2lr2hTm3kgnvB1lU9anXEaJoPxMCdtdH"
access_token = "145916665-EWEaAZsNq0VAOfGqKI5tPyDLcOD1SzzOrwStadiJ"
access_token_secret = "YAOzvQQziom4i30VnvlteJjDYMxpXdCRDalrRa5HciXhM"

# Tweepy 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target User Account
target_user = ("@BBC", "@CBS", "@CNN", "@FoxNews", "@nytimes")

# Variables for holding sentiments
sentiment_df = []

print(" --- begin request ---")

tweet_counter = 1

for user in target_user:
    # Loop through 10 pages of tweets (total 200 tweets)
    
    print("extracting tweets from %s"%user)
    
    for x in range(5):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user,page=x)

        # Loop through all tweets
        for tweet in public_tweets:

            # Run Vader Analysis on each tweet
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]

            sentiment_df.append({"Media":user,
                    "Tweet Text":tweet["text"],
                    "Compound":compound,
                    "Positive":pos,
                    "Negative":neg,
                    "Neutral":neu,
                    "Date":tweet["created_at"],
                    "Tweet Count":tweet_counter})
            
            tweet_counter = tweet_counter + 1 
                     
print("--- end request ---")
```

     --- begin request ---
    extracting tweets from @BBC
    extracting tweets from @CBS
    extracting tweets from @CNN
    extracting tweets from @FoxNews
    extracting tweets from @nytimes
    --- end request ---



```python
# create Tweet/Sentiment DF 

sentiment_final = pd.DataFrame.from_dict(sentiment_df)
sentiment_final
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Media</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet Count</th>
      <th>Tweet Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1280</td>
      <td>Wed Feb 14 20:03:04 +0000 2018</td>
      <td>@BBC</td>
      <td>0.118</td>
      <td>0.742</td>
      <td>0.140</td>
      <td>1</td>
      <td>💡 What does it take for kids from disadvantage...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4767</td>
      <td>Wed Feb 14 19:07:49 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.866</td>
      <td>0.134</td>
      <td>2</td>
      <td>RT @BBCWales: Heartwarming. ❤️\n\nCyril Jenkin...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.2886</td>
      <td>Wed Feb 14 19:07:35 +0000 2018</td>
      <td>@BBC</td>
      <td>0.171</td>
      <td>0.709</td>
      <td>0.120</td>
      <td>3</td>
      <td>RT @bbcrb: Isn't this a great idea? 💡💡\nNina u...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>Wed Feb 14 19:06:34 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>4</td>
      <td>RT @bbccomedy: 5 Reasons why Valentine's shoul...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.3612</td>
      <td>Wed Feb 14 19:00:07 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.815</td>
      <td>0.185</td>
      <td>5</td>
      <td>Chinese characters are based on symbols denoti...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0000</td>
      <td>Wed Feb 14 17:57:02 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>6</td>
      <td>What does it takes to survive in the most extr...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0000</td>
      <td>Wed Feb 14 17:28:27 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>7</td>
      <td>RT @bbcthesocial: | @LaurenAviah &amp;amp; @RickyC...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
      <td>Wed Feb 14 17:01:02 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>8</td>
      <td>❤️️🦌 A handy guide to dating, if you're a deer...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0000</td>
      <td>Wed Feb 14 16:57:13 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>9</td>
      <td>RT @BBCSport: What a way to spend Valentine's ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0000</td>
      <td>Wed Feb 14 16:53:35 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>10</td>
      <td>RT @bbc5live: Xanax turned my daughter into an...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0000</td>
      <td>Wed Feb 14 16:53:10 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>11</td>
      <td>RT @bbcthree: Working in a restaurant on V Day...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0000</td>
      <td>Wed Feb 14 16:40:01 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>12</td>
      <td>RT @BBCSport: This was INCREDIBLE.\n\n#Pyeongc...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.7419</td>
      <td>Wed Feb 14 15:16:41 +0000 2018</td>
      <td>@BBC</td>
      <td>0.294</td>
      <td>0.583</td>
      <td>0.123</td>
      <td>13</td>
      <td>RT @BBCRadio4: 📞Date night?! We want to hear y...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.8999</td>
      <td>Wed Feb 14 14:42:08 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.500</td>
      <td>0.500</td>
      <td>14</td>
      <td>RT @bbccomedy: Celebrate this #ValentinesDay w...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0000</td>
      <td>Wed Feb 14 14:32:02 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>15</td>
      <td>🧀🤤 Easy-peasy cheesy sauce! https://t.co/xG02l...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.7717</td>
      <td>Wed Feb 14 13:57:42 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.729</td>
      <td>0.271</td>
      <td>16</td>
      <td>RT @bbccomedy: Roses are great\nBut poems are ...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.6369</td>
      <td>Wed Feb 14 13:42:18 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.792</td>
      <td>0.208</td>
      <td>17</td>
      <td>RT @BBCiPlayer: Despite a rough start to life,...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0000</td>
      <td>Wed Feb 14 13:33:04 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>18</td>
      <td>A closer look at animal relationships reveals ...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.8030</td>
      <td>Wed Feb 14 13:03:03 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.772</td>
      <td>0.228</td>
      <td>19</td>
      <td>☝️📱💘 More couples than ever before are meeting...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>Wed Feb 14 12:48:47 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>20</td>
      <td>RT @bbcpress: 🌍 @BBC announces three-step plan...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.1280</td>
      <td>Wed Feb 14 20:03:04 +0000 2018</td>
      <td>@BBC</td>
      <td>0.118</td>
      <td>0.742</td>
      <td>0.140</td>
      <td>21</td>
      <td>💡 What does it take for kids from disadvantage...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.4767</td>
      <td>Wed Feb 14 19:07:49 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.866</td>
      <td>0.134</td>
      <td>22</td>
      <td>RT @BBCWales: Heartwarming. ❤️\n\nCyril Jenkin...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-0.2886</td>
      <td>Wed Feb 14 19:07:35 +0000 2018</td>
      <td>@BBC</td>
      <td>0.171</td>
      <td>0.709</td>
      <td>0.120</td>
      <td>23</td>
      <td>RT @bbcrb: Isn't this a great idea? 💡💡\nNina u...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0000</td>
      <td>Wed Feb 14 19:06:34 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>24</td>
      <td>RT @bbccomedy: 5 Reasons why Valentine's shoul...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.3612</td>
      <td>Wed Feb 14 19:00:07 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.815</td>
      <td>0.185</td>
      <td>25</td>
      <td>Chinese characters are based on symbols denoti...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.0000</td>
      <td>Wed Feb 14 17:57:02 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>26</td>
      <td>What does it takes to survive in the most extr...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0000</td>
      <td>Wed Feb 14 17:28:27 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>27</td>
      <td>RT @bbcthesocial: | @LaurenAviah &amp;amp; @RickyC...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0000</td>
      <td>Wed Feb 14 17:01:02 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>28</td>
      <td>❤️️🦌 A handy guide to dating, if you're a deer...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.0000</td>
      <td>Wed Feb 14 16:57:13 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>29</td>
      <td>RT @BBCSport: What a way to spend Valentine's ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0000</td>
      <td>Wed Feb 14 16:53:35 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>30</td>
      <td>RT @bbc5live: Xanax turned my daughter into an...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>-0.4019</td>
      <td>Wed Feb 14 19:20:06 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.153</td>
      <td>0.847</td>
      <td>0.000</td>
      <td>471</td>
      <td>RT @JohnBranchNYT: My story on Lindsey Jacobel...</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.0772</td>
      <td>Wed Feb 14 19:10:01 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.072</td>
      <td>0.844</td>
      <td>0.084</td>
      <td>472</td>
      <td>For the first time since 1945, Ash Wednesday a...</td>
    </tr>
    <tr>
      <th>472</th>
      <td>-0.0258</td>
      <td>Wed Feb 14 19:05:40 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.171</td>
      <td>0.711</td>
      <td>0.118</td>
      <td>473</td>
      <td>Don’t like a court ruling? For state legislato...</td>
    </tr>
    <tr>
      <th>473</th>
      <td>-0.2500</td>
      <td>Wed Feb 14 19:00:13 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.083</td>
      <td>0.917</td>
      <td>0.000</td>
      <td>474</td>
      <td>RT @nytimesarts: A report from the Queens Muse...</td>
    </tr>
    <tr>
      <th>474</th>
      <td>0.0000</td>
      <td>Wed Feb 14 18:55:07 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>475</td>
      <td>Jamie Brewer and her understudy, Edward Barban...</td>
    </tr>
    <tr>
      <th>475</th>
      <td>-0.6808</td>
      <td>Wed Feb 14 18:45:14 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.212</td>
      <td>0.712</td>
      <td>0.077</td>
      <td>476</td>
      <td>We asked 30 experts to think big, but realisti...</td>
    </tr>
    <tr>
      <th>476</th>
      <td>0.0000</td>
      <td>Wed Feb 14 18:35:08 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>477</td>
      <td>"You have to either get divorced or work throu...</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0.0000</td>
      <td>Wed Feb 14 18:31:03 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>478</td>
      <td>What's Nordic combined skiing? Start with two ...</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.4767</td>
      <td>Wed Feb 14 18:20:03 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.860</td>
      <td>0.140</td>
      <td>479</td>
      <td>The "Black Panther" soundtrack is nearly as de...</td>
    </tr>
    <tr>
      <th>479</th>
      <td>0.0000</td>
      <td>Wed Feb 14 18:10:01 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>480</td>
      <td>The official portraits of Barack Obama and Mic...</td>
    </tr>
    <tr>
      <th>480</th>
      <td>0.3612</td>
      <td>Wed Feb 14 18:00:28 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.104</td>
      <td>0.734</td>
      <td>0.162</td>
      <td>481</td>
      <td>RT @peterbakernyt: Best advice @maggieNYT ever...</td>
    </tr>
    <tr>
      <th>481</th>
      <td>-0.7184</td>
      <td>Wed Feb 14 17:50:05 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.318</td>
      <td>0.682</td>
      <td>0.000</td>
      <td>482</td>
      <td>“You can ban a day, but you can’t stop people ...</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.0000</td>
      <td>Wed Feb 14 17:35:07 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>483</td>
      <td>Symbolically, the showdown between Japan and K...</td>
    </tr>
    <tr>
      <th>483</th>
      <td>-0.5106</td>
      <td>Wed Feb 14 17:25:11 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.191</td>
      <td>0.809</td>
      <td>0.000</td>
      <td>484</td>
      <td>First, @UpshotNYT made a complete list of thin...</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.0000</td>
      <td>Wed Feb 14 17:10:08 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>485</td>
      <td>RT @jennydeluxe: hi my name is jenna and i'm a...</td>
    </tr>
    <tr>
      <th>485</th>
      <td>0.0516</td>
      <td>Wed Feb 14 17:00:15 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.943</td>
      <td>0.057</td>
      <td>486</td>
      <td>President Trump once said he would sign any im...</td>
    </tr>
    <tr>
      <th>486</th>
      <td>0.0000</td>
      <td>Wed Feb 14 16:50:08 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>487</td>
      <td>The U.S. has indicated that it will seek to pl...</td>
    </tr>
    <tr>
      <th>487</th>
      <td>0.0000</td>
      <td>Wed Feb 14 16:40:05 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>488</td>
      <td>While you're deciding who should be your Valen...</td>
    </tr>
    <tr>
      <th>488</th>
      <td>0.7845</td>
      <td>Wed Feb 14 16:30:12 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.734</td>
      <td>0.266</td>
      <td>489</td>
      <td>To win his third gold medal, Shaun White pulle...</td>
    </tr>
    <tr>
      <th>489</th>
      <td>-0.8316</td>
      <td>Wed Feb 14 16:20:10 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.302</td>
      <td>0.698</td>
      <td>0.000</td>
      <td>490</td>
      <td>When did the White House learn about the Rob P...</td>
    </tr>
    <tr>
      <th>490</th>
      <td>-0.4019</td>
      <td>Wed Feb 14 16:10:05 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.130</td>
      <td>0.870</td>
      <td>0.000</td>
      <td>491</td>
      <td>One person was injured after a shooting at the...</td>
    </tr>
    <tr>
      <th>491</th>
      <td>0.3818</td>
      <td>Wed Feb 14 16:00:31 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.097</td>
      <td>0.728</td>
      <td>0.175</td>
      <td>492</td>
      <td>Pennsylvania's Supreme Court struck down the s...</td>
    </tr>
    <tr>
      <th>492</th>
      <td>0.5106</td>
      <td>Wed Feb 14 15:45:18 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.858</td>
      <td>0.142</td>
      <td>493</td>
      <td>RT @nytimesarts: The portraits of former Presi...</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0.6249</td>
      <td>Wed Feb 14 15:30:04 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.758</td>
      <td>0.242</td>
      <td>494</td>
      <td>"There was not a pressing comb or relaxer on s...</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0.0772</td>
      <td>Wed Feb 14 15:17:53 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.939</td>
      <td>0.061</td>
      <td>495</td>
      <td>Forecasts show that America’s fertility is lik...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>0.4019</td>
      <td>Wed Feb 14 15:15:46 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.690</td>
      <td>0.310</td>
      <td>496</td>
      <td>And yes, it is still cold. https://t.co/VdWB8t...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>0.6597</td>
      <td>Wed Feb 14 15:13:15 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.748</td>
      <td>0.252</td>
      <td>497</td>
      <td>North Korean cheerleaders remain the source of...</td>
    </tr>
    <tr>
      <th>497</th>
      <td>-0.2023</td>
      <td>Wed Feb 14 15:10:23 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.107</td>
      <td>0.893</td>
      <td>0.000</td>
      <td>498</td>
      <td>Another Alpine event was postponed at the Yong...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>0.5267</td>
      <td>Wed Feb 14 15:08:13 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.861</td>
      <td>0.139</td>
      <td>499</td>
      <td>Shaun White placed fourth at the 2014 Sochi Ol...</td>
    </tr>
    <tr>
      <th>499</th>
      <td>0.6597</td>
      <td>Wed Feb 14 15:03:39 +0000 2018</td>
      <td>@nytimes</td>
      <td>0.000</td>
      <td>0.795</td>
      <td>0.205</td>
      <td>500</td>
      <td>Chloe Kim’s father, a Korean immigrant, gave u...</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python
# export to CSV 
sentiment_final.to_csv('news_sentiment_complete.csv', index=False)
```


```python
# Store filepath in a variable
compound_df = "news_sentiment_complete.csv"

```


```python
compound_df = pd.read_csv(compound_df, encoding = "ISO-8859-1")
compound_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Media</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet Count</th>
      <th>Tweet Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.1280</td>
      <td>Wed Feb 14 20:03:04 +0000 2018</td>
      <td>@BBC</td>
      <td>0.118</td>
      <td>0.742</td>
      <td>0.140</td>
      <td>1</td>
      <td>ð¡ What does it take for kids from disadvant...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4767</td>
      <td>Wed Feb 14 19:07:49 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.866</td>
      <td>0.134</td>
      <td>2</td>
      <td>RT @BBCWales: Heartwarming. â¤ï¸\n\nCyril Je...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.2886</td>
      <td>Wed Feb 14 19:07:35 +0000 2018</td>
      <td>@BBC</td>
      <td>0.171</td>
      <td>0.709</td>
      <td>0.120</td>
      <td>3</td>
      <td>RT @bbcrb: Isn't this a great idea? ð¡ð¡\n...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>Wed Feb 14 19:06:34 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>4</td>
      <td>RT @bbccomedy: 5 Reasons why Valentine's shoul...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.3612</td>
      <td>Wed Feb 14 19:00:07 +0000 2018</td>
      <td>@BBC</td>
      <td>0.000</td>
      <td>0.815</td>
      <td>0.185</td>
      <td>5</td>
      <td>Chinese characters are based on symbols denoti...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# # BBC Dataframe 
bbc_pd = compound_df.loc[compound_df["Media"] == "@BBC",:]

# CBS DataFrame
cbs_pd = compound_df.loc[compound_df["Media"] == "@CBS",:]

# CNN DataFrame
cnn_pd = compound_df.loc[compound_df["Media"] == "@CNN",:]

# Fox DataFrame
fox_pd = compound_df.loc[compound_df["Media"] == "@FoxNews",:]

# NYtimes Dataframe 
nytimes_pd = compound_df.loc[compound_df["Media"] == "@CBS",:]
```


```python
# Compound Sentiment Scatter Plot 

# ---------------------------------

bbc_plt = plt.plot(np.arange(len(bbc_pd["Compound"])),
         bbc_pd["Compound"], marker="o", color="blue", linewidth=0,
         alpha=0.8)


cbs_plt = plt.plot(np.arange(len(cbs_pd["Compound"])),
         cbs_pd["Compound"], marker="o", color="red", linewidth=0,
         alpha=0.8)

cnn_plt = plt.plot(np.arange(len(cnn_pd["Compound"])),
         cnn_pd["Compound"], marker="o", color="orange", linewidth=0,
         alpha=0.8)

fox_plt = plt.plot(np.arange(len(fox_pd["Compound"])),
         fox_pd["Compound"], marker="o", color="green", linewidth=0,
         alpha=0.8)

nytimes_plt = plt.plot(np.arange(len(nytimes_pd["Compound"])),
         nytimes_pd["Compound"], marker="o", color="purple", linewidth=0,
         alpha=0.8)

# define legend handle formatting
circ_bbc = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor="blue")
circ_cbs= Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor="red")
circ_cnn = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor="orange")
circ_fox = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor="green")
circ_nytimes = Line2D([0], [0], linestyle="none", marker="o", alpha=0.75, markersize=10, markerfacecolor="purple")

# define x and y limits 

plt.ylim(-1, 1)
plt.xlim(105, -5)

# Print scatter plot w/ formatting and legend 

plt.title("Sentiment Analysis of Media Tweets (%s)" % (time.strftime("%x")), fontsize=12)
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.legend((circ_bbc, circ_cbs, circ_cnn, circ_fox, circ_nytimes), ("@BBC", "@CBS", "@CNN", "@FoxNews", "@nytimes"),
           numpoints=1, bbox_to_anchor = (1,1), title="News Sources")
plt.grid(True)
plt.show()

# save figure 

plt.savefig("Compound Sentiment Analysis of News Media Tweets.png", bbox='tight')
```


![png](output_9_0.png)



```python
# Filter the DataFrame down only to those columns to chart
sentiment_bar_df = compound_df[["Media","Compound"]]

# Set the index to be "State" so they will be used as labels
sentiment_bar_df = sentiment_bar_df.set_index("Media")

avg_compound_media = sentiment_bar_df.groupby(["Media"]).mean()["Compound"]

avg_compound_media
```




    Media
    @BBC        0.212647
    @CBS        0.334809
    @CNN       -0.165498
    @FoxNews   -0.135197
    @nytimes   -0.025888
    Name: Compound, dtype: float64




```python
# Bar chart for media polarity scores based on media outlet 
#----------------------------------------------------------

# Labels for media channels 
tags = ["@BBC", "@CBS", "@CNN", "@FoxNews", "@nytimes"]

# The colors of each media channel 
colors = ["blue", "red", "orange", "green", "purple"]

xaxis = np.arange(len(tags))
yaxis = avg_compound_media

plt.figure(figsize = (10,7))

plt.bar(xaxis, yaxis, color = colors, align="edge")

tick_locations = [value+.4 for value in xaxis]
plt.xticks(tick_locations, tags)
plt.ylim(-.5, .5)
plt.axhline(y=0, linestyle='-', linewidth = 1, color = "black", alpha = .6)
plt.ylabel("Twitter Polarity")
plt.title("Overall Media Sentiment based on Twitter (%s)" % (time.strftime("%x")), fontsize=10)
plt.savefig("Overall Media Sentiment Anlysis of Last 100 Tweets.png", bbox_inches = 'tight')
plt.show()
```


    <matplotlib.figure.Figure at 0x118f12f98>



![png](output_11_1.png)



```python
# Analysis - as of 02/14/2018 CNN, FoxNews, and the New York Times, are all showing negative compounded scores
# as of 11:24 p.m. EST, while BBC, and CBS showed positive scores. 
    
# The value of this sort of analysis is that we can identify media trends and then take the media specific data 
# (tweet text) and determine the nature of the negative news. For instance, perhaps CNN, Fox, and NYtimes are all 
# reporting negative tweets about the current administration, some sort of political/celebrity scandal. Conversely, 
# this sort of exercise can help identify positive trends. What's making the BBC and CBS so happy at this moment? 
# Are they promoting fluff pieces? For instance, is everyone jazzed about a new celebrity couple, baby, or 
# (regarding the BBC) a royal wedding? More importantly, why focus on these rather than whatever their peers are 
# reporting? 

# Just from a brief look at the tweet text pulled from BBC, there was an overwhelming majority of tweets 
# addressing Valentines Day, with a few negative/cautionary tweets about drug abuse, and other serious matters. 

# CBS's overwhelming positivity was particularly interesting. When I dug into the text themselves it showed a much 
# longer span of time, and ranged from topics such as: The Best Super Bowl commercials, to finding the perfect 
# gift for your valentine, and the winter olympics. In short, it seems that of the media outlets reviewed, 
# CBS tweeted a lot of "happy" news. 

# CNN, was focused on school shootings, and controversy concerning the current presidential administration. They were
# far more critical and the word choice more evocative. Fox also covered the school shooting, but the overall 
# tone of the tweets was tempered by president Trump's attempts to console the nation in light of the tragedy. 
# As it is widely known that Fox is his preferred news network, this makes sense that he would tweet, retweet, and 
# be quoted by the network. 

# The NYTimes had the most varied content amongst the media networks examined. They covered everything from Valentine's
# day, the winter Olympics, and museum exhibits, and still touched on the aforementioned shooting. 
# I suspect that the variety of the news helped raise the score, because it wasn't focused on purely on tragedy, or 
# fluff. 


    

```
