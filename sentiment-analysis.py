import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
def trainmodel():
    review=pd.read_csv('reviews.csv')
    review=review.rename(columns={'text':'review'},inplace=False)

    X = review.review
    y = review.polarity

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)
    vector = CountVectorizer(stop_words = 'english',lowercase=False)
    vector.fit(X_train)
    print(vector.vocabulary_)
    X_transformed = vector.transform(X_train)
    X_transformed.toarray()

    X_test_transformed = vector.transform(X_test)
    
    naivebayes=MultinomialNB()
    naivebayes.fit(X_transformed, y_train)
    saved_model = pickle.dumps(naivebayes)
    s = pickle.loads(saved_model)
    return s,vector
s,vector=trainmodel()

def predict(review1):
    vec = vector.transform([review1]).toarray()
    sentiment = str(list(s.predict(vec))[0]).replace('0', 'NEGATIVE').replace('1', 'POSITIVE')
    return sentiment   

   
st.header('Sentiment Analyser classifier')


review1=st.text_area("Enter the text")
if st.button ('Analyse'):
    st.write(predict(review1))
