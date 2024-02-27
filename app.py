import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
    text = text.lower()  # lower case
    text = nltk.word_tokenize(text)  # tokenization

    y = []
    for i in text:  # removing special characters
        if i.isalnum():  # alphanumeric
            y.append(i)

    text = y[:]  # used for cloning
    y.clear()

    for i in text:  # removing stopwords and punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]  # making it to base word by removing 'ing' for example
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message: ")

if st.button('Predict'):

    #1.preprocess
    transform_sms=transform_text(input_sms)
    #2.vectorize
    vector_input = tfidf.transform([transform_sms])
    #3.predict
    result = model.predict(vector_input)[0]
    #4.display
    if result == 1:
        st.header("Spam !!")
    else:
        st.header("Not Spam")
