import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

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
        train = train[:10000]
        y = y[:10000]
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

    vectorizer = TfidfVectorizer(max_features = 1000)
    df1_vector = vectorizer.fit_transform(df1)
    df2_vector = vectorizer.fit_transform(df2)
    X = np.abs(df1_vector - df2_vector)

    train_size = int(np.round(.7*train_length))

    X_train = X[:train_length][:train_size]
    X_valid = X[train_length:][train_size:]
    y_train = y[:train_size]
    y_valid = y[train_size:]
    X_test = X[train_length:]

    print(X_train.shape, y_train.shape,
          X_valid.shape, y_valid.shape,
          X_test.shape)
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    print('Accuracy score of xgb:', clf.score(X_valid, y_valid))

    if output:
        df_output = pd.DataFrame({'test_id': test_id,
                                  'is_duplicate': None})
        y_pred = clf.predict(X_test)
        print(y_pred[:10])
        pred_length = len(y_pred)
        df_output.ix[:pred_length-1, 'is_duplicate'] = y_pred
        df_output.to_csv('submission.csv',
                         index=False)


if __name__ == '__main__':
    run(run_test=True, output=True)

