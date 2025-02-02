# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder
# from scipy.sparse import hstack
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
# import numpy as np

# # Load dataset
# df = pd.read_excel('merged_data.xlsx')
# df = df.dropna()

# # Vectorize the 'kriteria_teknis' column
# vectorizer = TfidfVectorizer()
# X_text = vectorizer.fit_transform(df['kriteria_teknis'])

# # One-hot encode the company location
# encoder = OneHotEncoder()
# X_city = encoder.fit_transform(df[['preferensi_lokasi']])

# # Combine the text features and the encoded city features
# X = hstack([X_text, X_city])

# # Use the company names as the target
# y = df['nama_perusahaan']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the K-Nearest Neighbors model
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)

# # Predict on the test set
# y_pred = knn.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Akurasi model adalah: {accuracy * 100:.2f}%')

# # Save the model, vectorizer, and encoder
# joblib.dump(knn, 'knn_model_with_city.pkl')
# joblib.dump(vectorizer, 'vectorizer_with_city.pkl')
# joblib.dump(encoder, 'encoder_with_city.pkl')
# def hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis):
#     """
#     Fungsi ini menghitung skor kecocokan antara keterampilan teknis mahasiswa dan kriteria teknis perusahaan.
#     Skor dihitung berdasarkan jumlah keterampilan teknis yang cocok.
#     """
#     keterampilan_set = set(keterampilan_teknis.lower().split(', '))
#     kriteria_set = set(kriteria_teknis.lower().split(', '))
    
#     # Hitung jumlah keterampilan yang cocok
#     kecocokan = keterampilan_set.intersection(kriteria_set)
#     skor = len(kecocokan) / len(kriteria_set) * 100  # Persentase kecocokan
    
#     return skor, kecocokan

# # Periksa apakah preferensi lokasi mempengaruhi hasil prediksi
# def prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi):
#     keterampilan_vec = vectorizer.transform([keterampilan_teknis])
#     city_vec = encoder.transform([[preferensi_lokasi]])
#     X_new = hstack([keterampilan_vec, city_vec])

#     # Ambil 3 tetangga terdekat
#     distances, indices = knn.kneighbors(X_new, n_neighbors=3, return_distance=True)

#     prediksi = {}

#     for i in range(indices.shape[1]):
#         nama_perusahaan = y_train.iloc[indices[0][i]]
#         kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]
#         skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)

#         prediksi[nama_perusahaan] = (skor_kecocokan, list(keterampilan_cocok))

#         if len(prediksi) == 2:
#             break

#     return prediksi


# # Contoh prediksi
# keterampilan_teknis_baru = 'HTML, CSS, JavaScript, PHP, Laravel, CodeIgniter, Bootstrap, Tailwind, Kotlin, Java, Figma'
# preferensi_lokasi_baru = 'padang'
# prediksi = prediksi_perusahaan_terbaik(keterampilan_teknis_baru, preferensi_lokasi_baru)

# for perusahaan, (skor, keterampilan_cocok) in prediksi.items():
#     print(f'Perusahaan: {perusahaan}, Skor kecocokan: {skor:.2f}%, Keterampilan cocok: {", ".join(keterampilan_cocok)}')


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_excel('merged_data.xlsx')

# Menghapus baris dengan nilai NaN
df = df.dropna()

# Vectorize kolom 'kriteria_teknis' (mengubah teks menjadi vektor numerik)
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df['kriteria_teknis'])

# One-hot encode kolom 'preferensi_lokasi' (mengubah kota menjadi vektor biner)
encoder = OneHotEncoder()
X_city = encoder.fit_transform(df[['preferensi_lokasi']])

# Gabungkan fitur teks dan kota
X = hstack([X_text, X_city])

# Gunakan nama perusahaan sebagai target
y = df['nama_perusahaan']

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model
knn.fit(X_train, y_train)

# Prediksi pada testing set
y_pred = knn.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model adalah: {accuracy * 100:.2f}%')

# Simpan model, vectorizer, dan encoder ke file .pkl
joblib.dump(knn, 'knn_model_with_city.pkl')
joblib.dump(vectorizer, 'vectorizer_with_city.pkl')
joblib.dump(encoder, 'encoder_with_city.pkl')

print("Model, vectorizer, dan encoder berhasil disimpan.")


