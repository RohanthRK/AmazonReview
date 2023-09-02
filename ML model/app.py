import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import re
import nltk
nltk.download("stopwords")
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#processing
df1 = pd.read_csv('amazon_alexa.tsv',sep='\t')
def data_processing(text):
    text = text.lower()
    text = re.sub(r"http\S+www\S+|https\S+", '', text, flags= re.MULTILINE)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)
df1.verified_reviews = df1['verified_reviews'].apply(data_processing)
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data
df1['verified_reviews'] = df1['verified_reviews'].apply(lambda x: stemming(x))

vectorizer = CountVectorizer()
vectorizer.fit(df1['verified_reviews'])


# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index_1.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    arr = [(x) for x in request.form.values()]
    vector = vectorizer.transform(arr)
    mnb_predict = model.predict(vector)
    x=mnb_predict[0]
    if x==0:
        return render_template("index_1.html", prediction_text = "The given review is Negative")
    else:
        return render_template("index_1.html", prediction_text = "The given review is Positive ")
    # features_ = [x for x in request.form.values()]
    # prediction = model.predict(features)
    # return render_template("index.html", prediction_text = "The given review is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)