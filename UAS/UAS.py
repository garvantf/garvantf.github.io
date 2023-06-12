import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import altair as alt
from sklearn import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

st.set_page_config(page_title='UAS PENDATA')
st.markdown("<h1 style='text-align: center;'>UAS PENAMBANGAN DATA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Nama : Garvan Taufiqurrahman Fawwaz</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>NIM : 210411100062</p>", unsafe_allow_html=True)
st.write("---")







description, preprocessing, modeling, implementation = st.tabs(["Description", "Preprocessing", "Modeling", "Implementation"])


with description:
    st.write("# Deskripsi Dataset ")
    st.write("#### Dataset yang digunakan adalah dataset Iris, dapat dilihat pada tabel dibawah ini:")
    df = pd.read_csv('Iris.csv')
    st.dataframe(df)
    st.write("###### Sumber Dataset : https://www.kaggle.com/datasets/uciml/iris")
    st.write(" Dataset Iris digunakan di R.A. Makalah klasik Fisher tahun 1936, The Use of Multiple Measurements in Taxonomic Problems, dan juga dapat ditemukan di UCI Machine Learning Repository.Ini mencakup tiga spesies iris dengan masing-masing 50 sampel serta beberapa sifat tentang setiap bunga. Satu spesies bunga dapat dipisahkan secara linear dari dua lainnya, tetapi dua lainnya tidak dapat dipisahkan secara linear satu sama lain.Kolom dalam kumpulan data ini adalah: Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species. ")
    
with preprocessing:
    st.write("# Rumus Normalisasi data")
    st.image('rumus_normalisasi.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)

    st.write("# Normalisasi data iris")
    # Baca file CSV
    data = pd.read_csv('Iris.csv')

    # Mengambil fitur-fitur yang ingin dinormalisasi
    fitur = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

    # Normalisasi menggunakan Min-Max Scaler
    scaler = MinMaxScaler()
    fitur_normalisasi = scaler.fit_transform(fitur)

    # Gabungkan fitur yang sudah dinormalisasi dengan kolom target
    data_normalisasi = pd.DataFrame(fitur_normalisasi, columns=fitur.columns)
    data_normalisasi['Species'] = data['Species']

    st.write(data_normalisasi)

with modeling:
    # Fungsi untuk membaca dataset iris
    def load_iris_dataset():
        data = pd.read_csv('Iris.csv')
        return data

    # Fungsi untuk melakukan preprocessing data
    def preprocess_data(data):
        # Pisahkan fitur dan target
        fitur = data.drop('Species', axis=1)
        target = data['Species']

        # Lakukan penskalaan fitur
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(fitur)

        return scaled_features, target

    # Memuat dataset iris
    data = load_iris_dataset()

    # Melakukan preprocessing data
    scaled_features, target = preprocess_data(data)

    # Nilai X training dan Nilai X testing
    training, test, training_label, test_label = train_test_split(
        scaled_features, target, test_size=0.2, random_state=1)

    # Streamlit App
    st.title('Modeling Iris Dataset')

    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan digunakan untuk pengecekan akurasi:")
        naive = st.checkbox('Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighbors')
        destree = st.checkbox('Decision Tree')
        mlp_model = st.checkbox('Artificial Neural Network (MLP)')
        submitted = st.form_submit_button("Submit")

        if submitted:
            if naive:
                # Model Naive Bayes
                gaussian = GaussianNB()
                gaussian.fit(training, training_label)
                y_pred = gaussian.predict(test)
                gaussian_accuracy = accuracy_score(test_label, y_pred)
                st.write('Model Naive Bayes accuracy score: {:.2f}'.format(gaussian_accuracy))

            if k_nn:
                # Model K-Nearest Neighbors
                k = 3  # Jumlah tetangga terdekat
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(training, training_label)
                y_pred = knn.predict(test)
                knn_accuracy = accuracy_score(test_label, y_pred)
                st.write('Model K-Nearest Neighbors accuracy score: {:.2f}'.format(knn_accuracy))

            if destree:
                # Model Decision Tree
                dt = DecisionTreeClassifier()
                dt.fit(training, training_label)
                y_pred = dt.predict(test)
                dt_accuracy = accuracy_score(test_label, y_pred)
                st.write('Model Decision Tree accuracy score: {:.2f}'.format(dt_accuracy))

            if mlp_model:
                # Model Artificial Neural Network (Multi-Layer Perceptron)
                mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
                mlp.fit(training, training_label)
                y_pred = mlp.predict(test)
                mlp_accuracy = accuracy_score(test_label, y_pred)
                st.write('Model Artificial Neural Network accuracy score: {:.2f}'.format(mlp_accuracy))
                
with implementation:
    # Memuat dataset iris
    data = pd.read_csv('Iris.csv')

    # Pisahkan fitur dan target
    X = data.iloc[:, :4].values  # Use all four features
    y = data['Species']

    # Lakukan penskalaan fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

    # Membangun model Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Streamlit App
    st.title('Klasifikasi Spesies Iris')

    # Menampilkan form untuk input fitur
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0)

    # Membuat prediksi berdasarkan input fitur
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    predicted_species = nb.predict(input_data)

    # Menampilkan hasil prediksi
    st.subheader('Hasil Klasifikasi')
    st.write('Spesies Iris yang diprediksi: ', predicted_species[0])