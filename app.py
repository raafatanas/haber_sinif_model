import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# NLTK'nin gerektirdiği kaynakları indir
nltk.download('punkt')
nltk.download('wordnet')

# Kaydedilmiş modeli, TF-IDF vektörleştiriciyi ve etiket kodlayıcısını yükle
with open('lightgbm_model.pkl', 'rb') as file:
    lgb_model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# Streamlit başlığı
st.title('Haber Sınıflandırma Uygulaması')
st.write('Bu uygulama, haber metinlerini sınıflandırmak için makine öğrenimi modelini kullanır. Lütfen aşağıya bir haber metni girin ve "Tahmin Et" butonuna tıklayın.')

# Yan panel
st.sidebar.header('Uygulama Bilgileri')
st.sidebar.write('Bu uygulama, haber metinlerini belirli sınıflara ayırmak için LightGBM modelini kullanır.')
st.sidebar.write('Metin girdisini tokenize eder ve lemmatize eder, ardından TF-IDF vektörleştirici ile dönüştürür ve modeli kullanarak sınıf tahmininde bulunur.')

# Kullanıcıdan girdi al
user_input = st.text_area("Haber metnini girin:", height=250, placeholder="Buraya haber metnini girin...")

if st.button('Tahmin Et'):
    if user_input:
        # Girdiği metni ön işleme tabi tut
        tokens = word_tokenize(user_input)
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_text = ' '.join(lemmatized_tokens)

        # Metni TF-IDF vektörleştirici ile dönüştür
        text_tfidf = tfidf_vectorizer.transform([processed_text])

        # Model ile tahmin yap
        prediction = lgb_model.predict(text_tfidf)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        # Sonucu göster
        st.markdown(f"### Tahmin Edilen Sınıf: **{prediction_label}**")
    else:
        st.warning("Lütfen bir haber metni girin.")
else:
    st.info("Haber metnini girdikten sonra 'Tahmin Et' butonuna tıklayın.")

# Ek bilgiler ve öneriler
st.sidebar.subheader('Nasıl Kullanılır:')
st.sidebar.write('1. Ana metin alanına bir haber metni girin.')
st.sidebar.write("2. 'Tahmin Et' butonuna tıklayın.")
st.sidebar.write('3. Tahmin edilen sınıf, ana sayfada görüntülenecektir.')

st.sidebar.subheader('Örnek Haber Metinleri:')
st.sidebar.write('- Ekonomi: "Borsa İstanbul’da yükseliş hız kesmeden devam ediyor."')
st.sidebar.write('- Spor: "Fenerbahçe, Galatasaray’ı 2-1 mağlup etti."')
st.sidebar.write('- Sağlık: "Covid-19 aşı çalışmaları hızla ilerliyor."')
