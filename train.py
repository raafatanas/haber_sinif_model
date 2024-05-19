import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging
import numpy as np

# NLTK'nin gerektirdiği kaynakları indir
print("NLTK kaynaklarını indirme başladı...")
nltk.download('punkt')
nltk.download('wordnet')
print("NLTK kaynakları indirildi.")

# Veriyi oku
print("Veri setini okuma başladı...")
data = pd.read_csv('C:/Users/Anas Raafat/Desktop/dogal_dil_proje/data.csv')
print("Veri seti okundu.")

# Bağımsız değişken ve bağımlı değişken olarak ayır
print("Bağımsız ve bağımlı değişkenleri ayırma başladı...")
X = data['Haber Gövdesi']
y = data['Sınıf']
print("Bağımsız ve bağımlı değişkenler ayrıldı.")

# Metni tokenlara bölmek için tokenizasyon yap
print("Tokenizasyon işlemi başladı...")
X = X.apply(word_tokenize)
print("Tokenizasyon işlemi tamamlandı.")

# Lemmatizasyon yap
print("Lemmatizasyon işlemi başladı...")
lemmatizer = WordNetLemmatizer()
X = X.apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
print("Lemmatizasyon işlemi tamamlandı.")

# Tekrar metin haline getir
print("Metni tekrar oluşturma işlemi başladı...")
X = X.apply(lambda tokens: ' '.join(tokens))
print("Metni tekrar oluşturma işlemi tamamlandı.")

# Eğitim ve test setlerini oluştur
print("Eğitim ve test setlerini ayırma işlemi başladı...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Eğitim ve test setleri ayrıldı.")

# TF-IDF vektörleştirici oluştur
print("TF-IDF vektörleştirici oluşturma işlemi başladı...")
tfidf_vectorizer = TfidfVectorizer(max_features=20000)
print("TF-IDF vektörleştirici oluşturuldu.")

# Eğitim ve test verilerini TF-IDF vektörlerine dönüştür
print("Eğitim ve test verilerini vektörlere dönüştürme işlemi başladı...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("Veriler vektörlere dönüştürüldü.")

# Sınıf etiketlerini sayısal değerlere dönüştür
print("Sınıf etiketlerini sayısal değerlere dönüştürme işlemi başladı...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
print("Sınıf etiketleri sayısal değerlere dönüştürüldü.")

# LightGBM modelini oluştur
print("LightGBM modeli oluşturma işlemi başladı...")
lgb_model = lgb.LGBMClassifier(device='gpu')
print("LightGBM modeli oluşturuldu.")

# LightGBM loglarını etkinleştir
logging.basicConfig(level=logging.INFO)

# Custom evaluation functions
def lgb_precision_score(y_true, y_pred):
    y_pred_max = [np.argmax(line) for line in y_pred]
    return 'precision', precision_score(y_true, y_pred_max, average='macro', zero_division=1), True

def lgb_recall_score(y_true, y_pred):
    y_pred_max = [np.argmax(line) for line in y_pred]
    return 'recall', recall_score(y_true, y_pred_max, average='macro', zero_division=1), True

def lgb_f1_score(y_true, y_pred):
    y_pred_max = [np.argmax(line) for line in y_pred]
    return 'f1', f1_score(y_true, y_pred_max, average='macro', zero_division=1), True

# Placeholder lists for metrics
train_precision = []
val_precision = []
train_recall = []
val_recall = []
train_f1 = []
val_f1 = []

# Modeli eğit
print("Model eğitme işlemi başladı...")
eval_set = [(X_train_tfidf, y_train_encoded), (X_test_tfidf, y_test_encoded)]
lgb_model.fit(
    X_train_tfidf, y_train_encoded,
    eval_set=eval_set,
    eval_metric=['multi_error', lgb_precision_score, lgb_recall_score, lgb_f1_score]
)
print("Model eğitildi.")

# Doğruluk skoru hesaplama işlemi başladı
print("Doğruluk skoru hesaplama işlemi başladı...")
y_pred_train = lgb_model.predict(X_train_tfidf)
y_pred_val = lgb_model.predict(X_test_tfidf)

train_accuracy = accuracy_score(y_train_encoded, y_pred_train)
val_accuracy = accuracy_score(y_test_encoded, y_pred_val)

print("Eğitim seti doğruluğu:", train_accuracy)
print("Test seti doğruluğu:", val_accuracy)

# Custom metric storage
for i in range(len(lgb_model.evals_result_['training']['multi_error'])):
    train_precision.append(precision_score(y_train_encoded, lgb_model.predict(X_train_tfidf, num_iteration=i+1), average='macro', zero_division=1))
    val_precision.append(precision_score(y_test_encoded, lgb_model.predict(X_test_tfidf, num_iteration=i+1), average='macro', zero_division=1))
    train_recall.append(recall_score(y_train_encoded, lgb_model.predict(X_train_tfidf, num_iteration=i+1), average='macro', zero_division=1))
    val_recall.append(recall_score(y_test_encoded, lgb_model.predict(X_test_tfidf, num_iteration=i+1), average='macro', zero_division=1))
    train_f1.append(f1_score(y_train_encoded, lgb_model.predict(X_train_tfidf, num_iteration=i+1), average='macro', zero_division=1))
    val_f1.append(f1_score(y_test_encoded, lgb_model.predict(X_test_tfidf, num_iteration=i+1), average='macro', zero_division=1))

# Hata oranını doğruluğa çevir
train_accuracy_metrics = [1 - x for x in lgb_model.evals_result_['training']['multi_error']]
val_accuracy_metrics = [1 - x for x in lgb_model.evals_result_['valid_1']['multi_error']]

# Accuracy grafiği
plt.figure(figsize=(10, 6))
plt.plot(train_accuracy_metrics, label='Train Accuracy')
plt.plot(val_accuracy_metrics, label='Validation Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Precision grafiği
plt.figure(figsize=(10, 6))
plt.plot(train_precision, label='Train Precision')
plt.plot(val_precision, label='Validation Precision')
plt.xlabel('Iteration')
plt.ylabel('Precision')
plt.title('Training and Validation Precision')
plt.legend()
plt.show()

# Recall grafiği
plt.figure(figsize=(10, 6))
plt.plot(train_recall, label='Train Recall')
plt.plot(val_recall, label='Validation Recall')
plt.xlabel('Iteration')
plt.ylabel('Recall')
plt.title('Training and Validation Recall')
plt.legend()
plt.show()

# F1-score grafiği
plt.figure(figsize=(10, 6))
plt.plot(train_f1, label='Train F1-score')
plt.plot(val_f1, label='Validation F1-score')
plt.xlabel('Iteration')
plt.ylabel('F1-score')
plt.title('Training and Validation F1-score')
plt.legend()
plt.show()

# Define the file path to save the model
model_file_path = 'lightgbm_model.pkl'

# Save the model to file
with open(model_file_path, 'wb') as file:
    pickle.dump(lgb_model, file)

print("Model saved successfully.")

# TF-IDF vektörleştiriciyi kaydet
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

# Sınıf etiketi kodlayıcısını kaydet
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)