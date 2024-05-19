import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Haberi sınıflandırmak için girdi al
input_news = input("Sınıflandırmak istediğiniz haber metnini girin: ")

# LightGBM modelini yükleyin
with open('lightgbm_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# TF-IDF vektörleştiriciyi yükle
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Sınıf etiketlerini yükle
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Metni tokenlara bölmek için tokenizasyon yap
input_news_tokens = word_tokenize(input_news)

# Lemmatizasyon yap
lemmatizer = WordNetLemmatizer()
input_news_tokens = [lemmatizer.lemmatize(word) for word in input_news_tokens]

# Tekrar metin haline getir
input_news_text = ' '.join(input_news_tokens)

# Haberi TF-IDF vektörüne dönüştür
input_news_tfidf = tfidf_vectorizer.transform([input_news_text])

# Haberin sınıfını tahmin et
predicted_class_encoded = loaded_model.predict(input_news_tfidf)

# Sınıf etiketini orijinal metne dönüştür
predicted_class = label_encoder.inverse_transform(predicted_class_encoded)

print("\n\nTahmin edilen haber sınıfı:", predicted_class[0])
