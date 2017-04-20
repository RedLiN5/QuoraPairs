import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb


df = pd.read_table('train.csv', header=0, index_col=0, sep=',')
test = pd.read_table('/Users/Leslie/GitHub/QuoraPairs/test.csv', header=0, index_col=None, sep=',')

df = df.fillna("")
test = test.fillna("")

df1 = df[['qid1', 'question1']]
df2 = df[['qid2', 'question2']]

