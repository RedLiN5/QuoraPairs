import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score

def run(run_test = False, output=False):
    train = pd.read_table('train.csv', header=0, index_col=0, sep=',')
    y = train['is_duplicate']
    train.drop(['qid1', 'qid2', 'is_duplicate'],
               axis=1, inplace=True)
    train_length = train.shape[0]

    test = pd.read_table('/Users/Leslie/GitHub/QuoraPairs/test.csv', header=0, index_col=None, sep=',')
    test_id = test['test_id']
    test.drop('test_id',
              axis=1, inplace=True)

    if run_test:
        train = train[:50000]
        y = y[:50000]
        train_length = train.shape[0]
        test = test[:10000]

    df = pd.concat([train, test])

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

    df1 = df['question1']
    df2 = df['question2']
    df_length = df1.shape[0]

    vectorizer = TfidfVectorizer(max_features = 5000)
    df_q1q2 = pd.concat([df1, df2])
    X_q1q2 = vectorizer.fit_transform(df_q1q2)
    transformer = TfidfTransformer(smooth_idf=False)
    X_q1q2 = transformer.fit_transform(X_q1q2)
    X_q1 = X_q1q2[:df_length]
    X_q2 = X_q1q2[df_length:]
    X = X_q1 - X_q2
    print('X Size:', X.shape)

    train_size = int(np.round(.7*train_length))

    X_train = X[:train_length][:train_size]
    X_valid = X[:train_length][train_size:]
    y_train = y[:train_size]
    y_valid = y[train_size:]
    X_test = X[train_length:]
    print('X_train:', X_train.shape,
          'X_valid:', X_valid.shape,
          'y_train:', y_train.shape,
          'y_valid:', y_valid.shape)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    valid_proba = clf.predict_proba(X_valid)
    print('AUC of xgb:', roc_auc_score(y_valid,
                                       valid_proba.T[1]))

    nb = BernoulliNB()
    nb.fit(X_train, y_train)
    valid_proba = nb.predict_proba(X_valid)
    print('AUC of NB:', roc_auc_score(y_valid,
                                      valid_proba.T[1]))

    if output:
        df_output = pd.DataFrame({'test_id': test_id,
                                  'is_duplicate': None})
        df_output = df_output[['test_id', 'is_duplicate']]
        pred_proba = nb.predict_proba(X_test).T[1]
        pred_length = len(pred_proba)
        df_output.ix[:pred_length-1, 'is_duplicate'] = pred_proba
        df_output.to_csv('submission.csv',
                         index=False)


if __name__ == '__main__':
    run(run_test=False,
        output=True)

