import pandas as pd
import streamlit as st
review = pd.read_csv("reviews.csv")
review = review.rename(columns = {"text": "review"},inplace = False)
from sklearn.model_selection import train_test_split
X = review.review
y = review.polarity
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(stop_words = "english", lowercase = False)
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_transformed.toarray()
X_test_transformed = vector.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)
import pickle
saved_model = pickle.dumps(naivebayes)
s = pickle.loads(saved_model)
st.header('Sentiment Analyzer')
input = st.text("Enter the word", value="")
vec = vector.transform([input]).toarray()
if st.button("Analyse"):
    st.write(str(list(s.predict(vec))[0]).replace("0","NEGATIVE").replace("1","POSITIVE"))
