import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax, Adam, SGD, Adagrad ,RMSprop

new_title = '<h1 style="font-family:sans-serif; color:NAVY; font-size: 50px; align ="right">Classification of rice varieties using numeric and image data by Deep Neural Network techniques</h1>'
st.markdown(new_title, unsafe_allow_html=True)

st.sidebar.subheader("SELECT A MODEL OF YOUR CHOICE")
x = st.sidebar.selectbox(label = 'MODEL',options = ["ANN", "DNN", "VGG16"])

st.write("\n\n\n\n\n")
m = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">INTRODUCTION</h2>'
st.markdown(m , unsafe_allow_html = True)
st.markdown("The goal of this research is to create a non-destructive model that uses images of rice varieties to improve classification performance and to extract various properties of grain products using an image processing system. Then, to identify raw photos, build deep learning models, compare them, and evaluate the findings.\n ")

m2 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">SAMPLE DATASET</h2>'
st.markdown(m2 , unsafe_allow_html = True)
imgsd = Image.open("sampledataset.png")
st.image(imgsd)
st.markdown("The dataset includes 5 different rice varities namely Arborio, Basmati, Ipsala, Jasmine and Karacadag")
st.markdown("Each variety contains 15,000 images. The whole dataset contains 75,000 images of sample data")

m2 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">DATA PREPROCESSING</h2>'
st.markdown(m2 , unsafe_allow_html = True)
st.markdown("To balance the data, Nan values where removed and then applied Standard Scalar and Principle component analysis")
#imgsd = Image.open("dataprep3.png")
#st.image(imgsd)

m2 = '<h2 style="font-family:sans-serif; color: BLACK; font-size: 20px; align ="right">In the project 3 different models were compared namely ANN, DNN, VGG16</h2>'
st.markdown(m2 , unsafe_allow_html = True)
m2 = '<h2 style="font-family:sans-serif; color: BLACK; font-size: 20px; align ="right">The proposed model is VGG16</h2>'
st.markdown(m2 , unsafe_allow_html = True)

df_labels = {
    'arborio' : 0,
    'basmati' : 1,
    'ipsala' : 2,
    'jasmine' : 3,
    'karacadag': 4
}
new_key = list(df_labels)

if x == "ANN":
    st.write("User Input")
    m3 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">MODEL DETAILS</h2>'
    st.markdown(m3 , unsafe_allow_html = True)
    if st.button("Click here to see model details"):
        st.write("Training Loss:    0.001284547965042293")
        st.write("Validation Loss:  0.004056261386722326")
        st.write("Train Score:      0.9998332858085632")
        st.write("Test Score:       0.999133288860321")
        img1 = Image.open("ANNresults.png")
        img2 = Image.open("ANN_classreport.png")
        img3 = Image.open("ANN_confusion.png")
        st.subheader("Model accuracy plots")
        st.image(img1)
        st.subheader("Confusion Matrix")
        st.image(img3)
        st.subheader("Classification Report")
        st.image(img2)

if x == "DNN":
    m3 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">MODEL DETAILS</h2>'
    st.markdown(m3 , unsafe_allow_html = True)
    if st.button("Click here to see model details"):
        st.write("Training Loss:    0.005439203232526779")
        st.write("Validation Loss:  0.07092532515525818")
        st.write("Train Score:      0.9993665814399719")
        st.write("Test Score:       0.9989332556724548")
        img1 = Image.open("DNNresults.png")
        img2 = Image.open("CNN_classification.png")
        img3 = Image.open("DNN_confusion.png")
        st.subheader("Model accuracy plots")
        st.image(img1)
        st.subheader("Classification Report")
        st.subheader("Confusion Matrix")
        st.image(img3)
        st.subheader("Classification Report")
        st.image(img2)

if x == "VGG16":
    m3 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">MODEL DETAILS</h2>'
    st.markdown(m3 , unsafe_allow_html = True)
    if st.button("Click here to see model details"):
        st.write("Training Loss: 0.02162024166584015")
        st.write("Validation Loss: 0.09527199065065384")
        st.write("Train Score: 0.999799943447113")
        st.write("Test Score: 0.9989452915256348")
        img1 = Image.open("CNNresults.png")
        img2 = Image.open("ANN_classreport.png")
        img3 = Image.open("ANN_confusion.png")
        st.subheader("Model accuracy plots")
        st.image(img1)
        st.subheader("Classification Report")
        st.subheader("Confusion Matrix")
        st.image(img3)
        st.subheader("Classification Report")
        st.image(img2)

m2 = '<h2 style="font-family:sans-serif; color: BROWN; font-size: 30px; align ="right">PREDICTION:</h2>'
st.markdown(m2 , unsafe_allow_html = True)
if x == "ANN":
  uploaded_file = st.file_uploader("Upload spreadsheet", type=["csv", "xlsx"])
  if uploaded_file:
    # Check MIME type of the uploaded file
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    label_encoder = preprocessing.LabelEncoder()
    df['CLASS']= label_encoder.fit_transform(df['CLASS'])
    df = df.dropna()
    X=df.drop(['CLASS'],axis=1)
    y=df.CLASS
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA().fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    model = load_model('/content/drive/MyDrive/Mini prj/Code/ANN_model.h5')
    prob = model.predict(X_test)
    labels = prob.argmax(axis=1)
    st.write("Your accuracy for the given dataset is: ")
    st.write(accuracy_score(y_test,labels))
    score_mse_test = model.evaluate(X_test, y_test)
    st.write('Test Score:', score_mse_test[1])
    score_mse_train = model.evaluate(X_train, y_train)
    st.write('Train Score:', score_mse_train[1])

if x == "DNN":
  uploaded_file = st.file_uploader("Upload spreadsheet", type=["csv", "xlsx"])
  if uploaded_file:
    # Check MIME type of the uploaded file
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    label_encoder = preprocessing.LabelEncoder()
    df['CLASS']= label_encoder.fit_transform(df['CLASS'])
    df = df.dropna()
    X=df.drop(['CLASS'],axis=1)
    y=df.CLASS
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA().fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    model = load_model('/content/drive/MyDrive/Mini prj/Code/DNN_model.h5')
    prob = model.predict(X_test)
    labels = prob.argmax(axis=1)
    st.write("Your accuracy for the given dataset is: ")
    st.write(accuracy_score(y_test,labels))
    score_mse_test = model.evaluate(X_test, y_test)
    st.write('Test Score:', score_mse_test[1])
    score_mse_train = model.evaluate(X_train, y_train)
    st.write('Train Score:', score_mse_train[1])

if x=='VGG16':
  img = st.file_uploader(label = "")
  if st.button("Predict"):
        if img is not None:
            im = Image.open(img)
            im = im.resize((256,256))
            pix_val = list(im.getdata())
            pix_val_flat = [x for sets in pix_val for x in sets]
            Data = np.array(pix_val_flat).reshape(-1,256,256,3)
            model3 = load_model("my_model.h5")
            ans = model3.predict(Data)[0]
            #st.write("CONFIDENCE =",max(ans)*100, "%")
            st.write("Prediction: ")
            index = ans.argmax(axis = 0)
            st.write(new_key[index])

st.sidebar.markdown("Developed by: ")
st.sidebar.markdown("123018040- Jaganath Sankaranarayanan")
st.sidebar.markdown("123018062- Nakendraprasath K")
st.sidebar.markdown("123018092- Sravan Srinivasan S")
