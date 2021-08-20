#import libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#importing smsspam dataset
data = pd.read_csv('smsspam.csv',sep='\t',names=['label','message'])

#data cleaning and pre-processing
data['label']=data['label'].map({'ham':0,'spam':1})
ps = PorterStemmer()
corpus = []
for i in range(len(data)):
  review = re.sub('{^a-zA-Z}',' ',data['message'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)
 
#building bag-of-words model for nlp
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()

#creating train and test dataset
X_train = x[:3000]
y_train = data['label'][:3000]
X_test = x[3000:]
y_test = data['label'][3000:]

#training with svm classifier
spam_svm = svm.SVC(C=1000)
spam_svm.fit(X_train, y_train)
y_pred_spam_svm = spam_svm.predict(X_test)

#different test score after training with svm classifier
svm.score(X_test, y_test)
f1_svm = f1_score(y_test,y_pred_spam_svm)
confusion_svm = confusion_matrix(y_test,y_pred_spam_svm)
accuracy_svm = accuracy_score(y_test,y_pred_spam_svm)

#training with logistic regression classifier
lreg = LogisticRegression()
lreg.fit(X_train,y_train)
y_pred_spam_lreg = spam_lreg.predict(X_test)

#test score after training with logistic regression classifier
lreg.score(X_test,y_test)
f1_lreg = f1_score(y_test,y_pred_spam_lreg)
confusion_lreg = confusion_matrix(y_test,y_pred_spam_lreg)
accuracy_lreg = accuracy_score(y_test,y_pred_spam_lreg)

#training with naive bayes classifier
nb = MultinomialNB()
nb.fit(X_train,y_train)
y_pred_spam_nb = spam_nb.predict(X_test)

#test score after training with naive bayes classifier
nb.score(X_test,y_test)
f1_nb = f1_score(y_test,y_pred_spam_nb)
confusion_nb = confusion_matrix(y_test,y_pred_spam_nb)
accuracy_nb = accuracy_score(y_test,y_pred_spam_nb)






