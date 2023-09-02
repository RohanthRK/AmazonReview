import pandas as pd
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
import pickle


df = pd.read_csv('amazon_alexa.tsv',sep='\t')
# df.head()

def data_processing(text):
    text = text.lower()
    text = re.sub(r"http\S+www\S+|https\S+", '', text, flags= re.MULTILINE)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


df.verified_reviews = df['verified_reviews'].apply(data_processing)


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

df['verified_reviews'] = df['verified_reviews'].apply(lambda x: stemming(x))

pos_reviews = df[df.feedback == 1]
# pos_reviews.head()

neg_reviews = df[df.feedback==0]
# neg_reviews.head()


X = df['verified_reviews']
Y = df['feedback']

cv = CountVectorizer()
X = cv.fit_transform(df['verified_reviews'])

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# print("Size of x_train: ",(x_train.shape))
# print("Size of y_train: ",(y_train.shape))
# print("Size of x_test: ",(x_test.shape))
# print("Size of y_test: ",(y_test.shape))

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

mnb = MultinomialNB()
mnb.fit(x_train, y_train)

# mnb_pred = mnb.predict(x_test)
# mnb_acc = accuracy_score(mnb_pred, y_test)

# print("Test accuracy: {:.2f}%".format(mnb_acc*100))
# print(confusion_matrix(y_test, mnb_pred))
# print("\n")
# print(classification_report(y_test, mnb_pred))


pickle.dump(mnb,open("model.pkl","wb"))


# input

# n=input()
# arr=[n]
# vectorizer = CountVectorizer()
# vectorizer.fit(df['verified_reviews'])
# vector = vectorizer.transform(arr)
# mnb_predict = mnb.predict(vector)
# x=mnb_predict[0]
# print(x)
# if x==0:
#     print("It is a negative review.")
# else:
#     print("It is a Positive review")
    

