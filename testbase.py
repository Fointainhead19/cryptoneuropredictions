import nltk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
review = pd.read_csv("/kaggle/input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
consumer = pd.read_csv("/kaggle/input/consumer-reviews-of-amazon-products/1429_1.csv")
review19 = pd.read_csv("/kaggle/input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

review = review[['id','name','categories','reviews.date','reviews.doRecommend','reviews.numHelpful','reviews.rating','reviews.text','reviews.title']]
consumer = consumer[['id','name','categories','reviews.date','reviews.doRecommend','reviews.numHelpful','reviews.rating','reviews.text','reviews.title']]
review = review.append(consumer)
review19 = review19[['id','name','categories','reviews.date','reviews.doRecommend','reviews.numHelpful','reviews.rating','reviews.text','reviews.title']]
review = review.append(review19)

from nltk.corpus import stopwords
print(stopwords.words('english'))

for i in range(0,len(review)):
    if (review.at[i,'reviews.rating']==5):
        review['reviews.rating'][i] = "pos"
    else:
        review['reviews.rating'][i] = 'neg'
print(review['reviews.rating'])

print(len(review[review['reviews.rating']=='pos']))
print(len(review[review['reviews.rating']=='neg']))

review_balance = pd.DataFrame()
#We will be taking 500 samples from each outcome
review_balance = review_balance.append(review[review['reviews.rating']=='pos'].sample(n=3000,random_state=8))
#print(len(review_balance))
review_balance = review_balance.append(review[review['reviews.rating']=='neg'].sample(n=3000,random_state=8))
#print(len(review_balance))
review_balance = review_balance.reset_index().drop(columns = ['index'])
print(review_balance)

from nltk.tokenize import word_tokenize
 
words = pd.DataFrame(columns = ['words'])

stop_words = set(stopwords.words('english'))

#change to review.index after complete, this helps with runtime
for num in review_balance.index:
    word_tokens = word_tokenize(review_balance['reviews.text'][num])
    word_tokens = [w.lower() for w in word_tokens]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            if w.isalpha():
                filtered_sentence.append(w)
    words = words.append({'words': filtered_sentence}, ignore_index=True)
review_balance['words'] = words
review_balance
#temp = temp.append({'Composite': comp}, ignore_index=True)

X_train,X_test,Y_train, Y_test = train_test_split(review_balance['words'], review_balance['reviews.rating'], test_size=0.25, random_state=30)
X_train = X_train.reset_index().drop(columns = ['index'])
X_test = X_test.reset_index().drop(columns = ['index'])
Y_train = Y_train.reset_index().drop(columns = ['index'])
Y_test = Y_test.reset_index().drop(columns = ['index'])

print(Y_train)
print(Y_test)
print(X_train)
print(X_test)

X_train.head(10)

from collections import Counter

#counts of all words
vocab = Counter()

for num in X_train.index:
    vocab.update(X_train['words'][num])
    

print(len(vocab))
# keep tokens with a min occurrence
min_occurances = 2
for k,c in list(vocab.items()):   # list is important here
    if (c < min_occurances) or (len(k)<2):
        del(vocab[k])

print(len(vocab))
print(vocab.most_common(5))
from sklearn.feature_extraction.text import TfidfVectorizer
count=0
X_train_string = pd.DataFrame(columns = ['strings'])
for words in X_train['words']:
    X_train_string = X_train_string.append({'strings': ' '.join(words)}, ignore_index=True)

X_test_string = pd.DataFrame(columns = ['strings'])
for words in X_test['words']:
    X_test_string = X_test_string.append({'strings': ' '.join(words)}, ignore_index=True)

vectorizer = TfidfVectorizer()
vectorizer = vectorizer.fit(X_train_string['strings'])
tf_x_train = vectorizer.transform(X_train_string['strings'])
tf_x_test = vectorizer.transform(X_test_string['strings'])

print(tf_x_train[0])
print(tf_x_test[0])

from sklearn import svm
model = svm.SVC(kernel='linear',random_state=8)
model.fit(tf_x_train,Y_train)
y_test_pred=model.predict(tf_x_test)
from sklearn.metrics import classification_report
report=classification_report(Y_test, y_test_pred)
print(report)
from sklearn import svm
model = svm.SVC(random_state=8)
model.fit(tf_x_train,Y_train)
y_test_pred=model.predict(tf_x_test)
from sklearn.metrics import classification_report
report=classification_report(Y_test, y_test_pred)
print(report)
grid = svm.SVC(C=1, gamma=1,random_state=8)
grid.fit(tf_x_train,Y_train)
grid_predictions = grid.predict(tf_x_test)
# print classification report
print(classification_report(grid_predictions,Y_test))
test = pd.DataFrame(columns=['string'])
#fill this with a review you want to determine the sentiment of
test = test.append({'string': ' '.join(['The product is usefull'])}, ignore_index=True)
example = vectorizer.transform(test['string'])
y_pred = grid.predict(example)
print(y_pred)
