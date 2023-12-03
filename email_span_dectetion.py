import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data=pd.read_csv(r"D:/face_emotions/spam.csv")
print(data)
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.groupby("Category").describe())
data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
print(data.head(5))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
print(clf.fit(X_train,y_train))
emails=[
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]
print(clf.predict(emails))
print(clf.score(X_test,y_test))