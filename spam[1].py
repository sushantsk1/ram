import nltk
from nltk.corpus import stopwords
import string
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
messages=pd.read_csv("spam.csv",encoding='latin-1')
messages.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1,inplace=True)
messages = messages.rename(columns={'v1': 'class','v2': 'text'})
messages.head()
messages.groupby('class').describe()
messages['length']=messages['text'].apply(len)
messages.hist(column='length',by='class',bins=50,figsize=(15,6))
def process_text(text):
    #1 Remove punctuation
    nopunc=[char for char in text if char not in string.punctuation]
    nopunc=''.join(nopunc)
    
    #2 Remove stopwords
    clean_words=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3 Return list of clean text words
    return clean_words
messages['text'].apply(process_text).head()
msg_train,msg_test,class_train,class_test =train_test_split(messages['text'],messages['class'],test_size=0.2)
pipeline=Pipeline([
    ('bow',CountVectorizer(analyzer=process_text)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
     
    ])
pipeline.fit(msg_train,class_train)
predictions = pipeline.predict(msg_test)
print(classification_report(class_test,predictions))
import seaborn as sns
sns.heatmap(confusion_matrix(class_test,predictions),annot=True)
plt.show()

