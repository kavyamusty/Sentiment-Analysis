# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 21:55:54 2019

@author: kavya
"""

from sklearn.model_selection import train_test_split
import re
import pandas as pd
import csv
import numpy as np
import sys
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import OrderedDict

labels=['instance_number', 'tweet_text', 'topic_id', 'sentiment', 'is_sarcastic']
tweet_data=pd.read_table(sys.argv[1],sep='\t',quoting=csv.QUOTE_NONE,names=labels)
test_data=pd.read_table(sys.argv[2],sep='\t',quoting=csv.QUOTE_NONE,names=labels)
tweet_data.drop(['instance_number','sentiment','is_sarcastic'],axis=1,inplace= True)
test_data.drop(['topic_id','is_sarcastic','sentiment'],axis=1,inplace= True)

def preprocess_tweet(tweet):
    tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',tweet)# replace URL with space
    return tweet
tweet_data['tweet_text']=tweet_data['tweet_text'].apply(lambda y:preprocess_tweet(y))
new_tweet_data=tweet_data['tweet_text'].copy()
from nltk.tokenize import RegexpTokenizer
Stemmed_tweet=[]
porter=PorterStemmer() #STEMMING
for i in range (new_tweet_data.size):
    tweet = new_tweet_data[i]
    tokenizer = RegexpTokenizer('[a-zA-Z0-9#@%_$]\w+')
    token_words = tokenizer.tokenize(tweet)
    stem_sentence =[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    new_tweet_data[i] = "".join(stem_sentence)

tweets_data=new_tweet_data 
count=CountVectorizer(max_features=2300,lowercase=True,token_pattern=r'[a-zA-Z0-9#@%_$]+[a-zA-Z0-9#@%_$]+',stop_words=stopwords.words('english'))#max_df=0.9,min_df=5,#max_features=2500,#,stop_words=stopwords.words('english'))#token_pattern=r'[a-zA-Z0-9#@%_$]+',lowercase=False
bag_of_words=count.fit_transform(tweets_data)
bag_of_words_2=count.transform(test_data['tweet_text'])

X=bag_of_words.toarray()
Y=np.array(tweet_data['topic_id'])

x_train=X
x_test=bag_of_words_2.toarray()
y_train=Y

from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report

clf = MultinomialNB()
model = clf.fit(x_train, y_train)

predictions=model.predict(x_test)
instance=test_data['instance_number']
dic=OrderedDict()
for i in range(len(instance)):
    dic[instance[i]]=predictions[i]

for k, v in dic.items():
    print(str(k) + ' '+ str(v))
