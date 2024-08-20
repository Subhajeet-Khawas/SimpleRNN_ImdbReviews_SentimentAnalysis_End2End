from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
import streamlit as st

model=load_model("SimpleRNN_imdb.h5")

word_index=imdb.get_word_index()
rev_word_index={v:k for k,v in word_index.items()}

def decode_review(encoded_review):
    return " ".join([rev_word_index.get(i-3,"?") for i in encoded_review])

def preprocess_input(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2) + 3 for word in words]
    padded_review=pad_sequences([encoded_review],maxlen=500)
    return padded_review

st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review to classify it as positive or negative.')
user_input=st.text_area("Movie Review")
if st.button("Classify"):
    processed_input=preprocess_input(user_input)
    prob=model.predict(processed_input)
    if prob[0][0]>0.5:
        sentiment="Positive"
    else:
        sentiment="Negative"
    st.write("Review Entered: {}".format(user_input))
    st.write("Sentiment is {}".format(sentiment))
    st.write("P(E) of Sentiment is {}".format(np.around(prob[0][0],2)))
else:
    st.write('Please enter a movie review.')




