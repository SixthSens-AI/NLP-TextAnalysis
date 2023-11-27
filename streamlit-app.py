import streamlit as st

st.set_page_config(
    page_title="NLP Text Analysis | SixthSens AI",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRytIkMwtRARPB7H2pnEQNqdBgONVK2P7eBjw&usqp=CAU",
    layout="wide",  # centered, wide
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
            <style>
            .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
            .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
            .viewerBadge_text__1JaDK {display: none;}
            MainMenu {visibility: hidden;}
            header { visibility: hidden; }
            footer {visibility: hidden;}
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import string


nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")

# Load the models and vectorizers
model1 = joblib.load('./model/spam_detection_model.pkl')
model2 = joblib.load('./model/sentiment_analysis_model.pkl')
model3 = joblib.load('./model/stress_detection_model.pkl')
model4 = joblib.load('./model/hate_offensive_content_model.pkl')
model5 = joblib.load('./model/sarcasm_detection_model.pkl')

tfidf1 = joblib.load('./model/tfidf/tfidf1.pkl')
tfidf2 = joblib.load('./model/tfidf/tfidf2.pkl')
tfidf3 = joblib.load('./model/tfidf/tfidf3.pkl')
tfidf4 = joblib.load('./model/tfidf/tfidf4.pkl')
tfidf5 = joblib.load('./model/tfidf/tfidf5.pkl')

@st.cache_data
def get_state():
    return {}


state = get_state()

#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

rad=st.sidebar.radio("Navigation",["Home","Spam or Ham Detection","Sentiment Analysis","Stress Detection","Hate and Offensive Content Detection","Sarcasm Detection"])

#Home Page
if rad=="Home":
    st.title("SixthSens AI Complete Text Analysis App")
    st.image("Complete Text Analysis Home Page.jpg")
    st.text(" ")
    st.text("The Following Text Analysis Options Are Available->")
    st.text(" ")
    st.text("1. Spam or Ham Detection")
    st.text("2. Sentiment Analysis")
    st.text("3. Stress Detection")
    st.text("4. Hate and Offensive Content Detection")
    st.text("5. Sarcasm Detection")

#Spam Detection Analysis Page
if rad=="Spam or Ham Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    sent1=st.text_area("Enter The Text")
    transformed_sent1=transform_text(sent1)
    vector_sent1=tfidf1.transform([transformed_sent1])
    prediction1=model1.predict(vector_sent1)[0]

    if st.button("Predict"):
        if prediction1=="spam":
            st.warning("Spam Text!!")
        elif prediction1=="ham":
            st.success("Ham Text!!")

#Sentiment Analysis Page
if rad=="Sentiment Analysis":
    st.header("Detect The Sentiment Of The Text!!")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent2=tfidf2.transform([transformed_sent2])
    prediction2=model2.predict(vector_sent2)[0]

    if st.button("Predict"):
        if prediction2==0:
            st.warning("Negative Text!!")
        elif prediction2==1:
            st.success("Positive Text!!")

#Stress Detection Page
if rad=="Stress Detection":
    st.header("Detect The Amount Of Stress In The Text!!")
    sent3=st.text_area("Enter The Text")
    transformed_sent3=transform_text(sent3)
    vector_sent3=tfidf3.transform([transformed_sent3])
    prediction3=model3.predict(vector_sent3)[0]

    if st.button("Predict"):
        if prediction3>=0:
            st.warning("Stressful Text!!")
        elif prediction3<0:
            st.success("Not A Stressful Text!!")

#Hate & Offensive Content Page
if rad=="Hate and Offensive Content Detection":
    st.header("Detect The Level Of Hate & Offensive Content In The Text!!")
    sent4=st.text_area("Enter The Text")
    transformed_sent4=transform_text(sent4)
    vector_sent4=tfidf4.transform([transformed_sent4])
    prediction4=model4.predict(vector_sent4)[0]

    if st.button("Predict"):
        if prediction4==0:
            st.exception("Highly Offensive Text!!")
        elif prediction4==1:
            st.warning("Offensive Text!!")
        elif prediction4==2:
            st.success("Non Offensive Text!!")

#Sarcasm Detection Page
if rad=="Sarcasm Detection":
    st.header("Detect Whether The Text Is Sarcastic Or Not!!")
    sent5=st.text_area("Enter The Text")
    transformed_sent5=transform_text(sent5)
    vector_sent5=tfidf5.transform([transformed_sent5])
    prediction5=model5.predict(vector_sent5)[0]

    if st.button("Predict"):
        if prediction5==1:
            st.warning("Sarcastic Text!!")
        elif prediction5==0:
            st.success("Non Sarcastic Text!!")
