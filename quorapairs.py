import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb


df = pd.read_table('train.csv', header=0, index_col=0, sep=',')
test = pd.read_table('/Users/Leslie/GitHub/QuoraPairs/test.csv', header=0, index_col=None, sep=',')

df.question1=df.question1.str.lower()
df.question2=df.question2.str.lower()
df.question1=df.question1.astype(str)
df.question2=df.question2.astype(str)

abbr_dict={
    "what's":"what is",
    "what're":"what are",
    "who's":"who is",
    "who're":"who are",
    "where's":"where is",
    "where're":"where are",
    "when's":"when is",
    "when're":"when are",
    "how's":"how is",
    "how're":"how are",

    "i'm":"i am",
    "we're":"we are",
    "you're":"you are",
    "they're":"they are",
    "it's":"it is",
    "he's":"he is",
    "she's":"she is",
    "that's":"that is",
    "there's":"there is",
    "there're":"there are",

    "i've":"i have",
    "we've":"we have",
    "you've":"you have",
    "they've":"they have",
    "who've":"who have",
    "would've":"would have",
    "not've":"not have",

    "i'll":"i will",
    "we'll":"we will",
    "you'll":"you will",
    "he'll":"he will",
    "she'll":"she will",
    "it'll":"it will",
    "they'll":"they will",

    "isn't":"is not",
    "wasn't":"was not",
    "aren't":"are not",
    "weren't":"were not",
    "can't":"can not",
    "couldn't":"could not",
    "don't":"do not",
    "didn't":"did not",
    "shouldn't":"should not",
    "wouldn't":"would not",
    "doesn't":"does not",
    "haven't":"have not",
    "hasn't":"has not",
    "hadn't":"had not",
    "won't":"will not",
    '["\'?,\.]':'',
    '\s+':' ', # replace multi space with one single space
}

df.replace(abbr_dict,regex=True,inplace=True)
df = df.fillna("")
test = test.fillna("")

df1 = df[['qid1', 'question1']]
df2 = df[['qid2', 'question2']]

