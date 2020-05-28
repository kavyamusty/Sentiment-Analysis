# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:21:36 2019

@author: kavya
"""

from sklearn.model_selection import train_test_split
import re
import pandas as pd
import numpy as np
import csv
import sys
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
from sklearn import tree
#from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report


labels=['instance_number', 'tweet_text', 'topic_id', 'sentiment', 'is_sarcastic']
train_data=pd.read_table(sys.argv[1],sep='\t',quoting=csv.QUOTE_NONE,names=labels)
test_data=pd.read_table(sys.argv[2],sep='\t',quoting=csv.QUOTE_NONE,names=labels)
train_data.drop(['instance_number','topic_id','is_sarcastic'],axis=1,inplace= True)
test_data.drop(['topic_id','is_sarcastic','sentiment'],axis=1,inplace= True)

def preprocess_tweet(tweet):
    tweet=re.sub('https?://[^\s]+',' ',tweet) # replace URL with space
    return tweet
train_data['tweet_text']=train_data['tweet_text'].apply(lambda y:preprocess_tweet(y))


def sentiment_analysis(tweet):
    sentiment=[]
    for line in tweet['sentiment']:
        sentiment.append(line)
    return sentiment

tweet_data=train_data['tweet_text']
sentiment=sentiment_analysis(train_data)
count=CountVectorizer(lowercase=False,token_pattern=r'[a-zA-Z0-9#@%_$]+[a-zA-Z0-9#@%_$]+')
bag_of_words=count.fit_transform(tweet_data)
bag_of_words_2=count.transform(test_data['tweet_text'])

X=bag_of_words.toarray()
Y=np.array(sentiment)

x_train=X
x_test=bag_of_words_2.toarray()
y_train=Y

clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0,min_samples_leaf=20,max_features=200)
model = clf.fit(x_train, y_train)

predictions=model.predict(x_test)

instance=test_data['instance_number']
dic=OrderedDict()
for i in range(len(instance)):
    dic[instance[i]]=predictions[i]

for k, v in dic.items():
    print(str(k) + ' '+ str(v))
