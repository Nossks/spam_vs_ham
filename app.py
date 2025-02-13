# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()

# import nltk
# nltk.download('punkt')

# def transform_text(text):
#     text=text.lower()
#     text=nltk.word_tokenize(text)
#     y=[]
#     for word in text:
#         if word.isalnum():
#             y.append(word)

#     text=y[:]
#     y.clear()

#     stop_words = set(stopwords.words("english"))
#     punctuation = set(string.punctuation)

#     for word in text:
#         if word not in stop_words and word not in punctuation:
#             y.append(word)

#     text=y[:]
#     y.clear()

#     ps=PorterStemmer()
#     for word in text:
#         y.append(ps.stem(word))

#     return " ".join(y)


# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Email Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):
#     transformed_sms = transform_text(input_sms)
#     vector_input = tfidf.transform([transformed_sms])
#     result = model.predict(vector_input)[0]
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

import nltk
import os
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure nltk data is downloaded at runtime (Streamlit Cloud)
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words("english"))
    punctuation = set(string.punctuation)

    for word in text:
        if word not in stop_words and word not in punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# Load saved models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
