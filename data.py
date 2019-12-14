# Installing and importing libraries



import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np
import re, string
from gensim.models.keyedvectors import KeyedVectors
# %matplotlib inline

# Loading the Data"""

tweets = pd.read_csv('sentiment_tweets3.csv')
tweets.head(20)

tweets.drop(['Unnamed: 0'], axis = 1, inplace = True)

tweets['label'].value_counts()

tweets.info()

"""# Splitting the Data in Training and Testing Sets

As you can see, I used almost all the data for training: 98% and the rest for testing.
"""

trainIndex, testIndex = list(), list()
for i in range(tweets.shape[0]):
    if np.random.uniform(0, 1) < 0.98:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = tweets.iloc[trainIndex]
testData = tweets.iloc[testIndex]

tweets.info()

trainData['label'].value_counts()

trainData.head()

testData['label'].value_counts()

testData.head()


x_train = []
y_train = []
x_test = []
y_test = []
x_train_str = ""
x_test_str = ""


for row in trainData.message:
    x_train.append(row)
for row in trainData.label:
    y_train.append(row)
for row in testData.message:
    x_test.append(row)
for row in testData.label:
    y_test.append(row)
for row in trainData.message:
    x_train_str += row
for row in testData.message:
    x_test_str += row
    


punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
def strip_punc(corpus):
    return punc_regex.sub('', corpus)

    
    
for post in x_train:
    post = strip_punc(post).lower()
for post in x_test:
    post = strip_punc(post).lower()
    
path = r"./glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

with open("./dat/stopwords.txt", 'r') as r:
    stops = []
    for line in r:
        stops += [i.strip() for i in line.split('\t')]

del tweets
del trainData
del testData
