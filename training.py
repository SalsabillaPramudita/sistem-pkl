# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
# from scipy.sparse import hstack
# import nltk

# # Memuat dataset
# df = pd.read_excel(r'D:/sistem prediksi pkl/backend/merged_data.xlsx')
# df = df.dropna()

# # Vektorisasi kolom 'kriteria_teknis' tanpa stemming dan stop words 'english'
# vectorizer = TfidfVectorizer(stop_words='english')
# X_text = vectorizer.fit_transform(df['kriteria_teknis'])

# # One-hot encoding untuk kolom 'preferensi_lokasi'
# encoder = OneHotEncoder()
# X_city = encoder.fit_transform(df[['preferensi_lokasi']])

# # Gabungkan fitur teks dan lokasi yang telah di-encode
# X = hstack([X_text, X_city])

# # Target prediksi (nama perusahaan)
# y = df['nama_perusahaan']

# # Membagi data menjadi data latih dan uji
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Melatih model K-Nearest Neighbors
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)

# # Prediksi dengan data uji
# y_pred = knn.predict(X_test)

# # Menghitung akurasi
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Akurasi model adalah: {accuracy * 100:.2f}%')

# # Simpan model, vectorizer, dan encoder
# joblib.dump(knn, 'knn_model_with_city.pkl')
# joblib.dump(vectorizer, 'vectorizer_with_city.pkl')
# joblib.dump(encoder, 'encoder_with_city.pkl')

# # Fungsi untuk menghitung skor kecocokan
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

# # One-hot encoding untuk kolom preferensi lokasi
# encoder = OneHotEncoder()
# X_city = encoder.fit_transform(df[['preferensi_lokasi']])


# # Fungsi prediksi untuk merekomendasikan perusahaan terbaik
# def prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi):
#     keterampilan_vec = vectorizer.transform([keterampilan_teknis])
#     city_vec = encoder.transform([[preferensi_lokasi]])
#     X_new = hstack([keterampilan_vec, city_vec])

#     # Ambil 5 tetangga terdekat dan jaraknya
#     distances, indices = knn.kneighbors(X_new, n_neighbors=5, return_distance=True)

#     prediksi = {}

#     for i in range(indices.shape[1]):
#         nama_perusahaan = y_train.iloc[indices[0][i]]
#         kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]

#         # Hitung kecocokan keterampilan teknis dengan kriteria teknis perusahaan
#         skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)

#         prediksi[nama_perusahaan] = (skor_kecocokan, list(keterampilan_cocok))

#         if len(prediksi) == 2:  # Hanya ambil 2 perusahaan terbaik
#             break

#     return prediksi

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from scipy.sparse import hstack

# Memuat dataset
df = pd.read_excel(r'D:/sistem prediksi pkl/backend/merged_data.xlsx')
df = df.dropna()

# Vektorisasi kolom 'kriteria_teknis' tanpa stemming dan stop words 'english'
vectorizer = TfidfVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(df['kriteria_teknis'])

# One-hot encoding untuk kolom 'preferensi_lokasi'
encoder = OneHotEncoder()
X_city = encoder.fit_transform(df[['preferensi_lokasi']])

# Gabungkan fitur teks dan lokasi yang telah di-encode
X = hstack([X_text, X_city])

# Target prediksi (nama perusahaan)
y = df['nama_perusahaan']

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediksi dengan data uji
y_pred = knn.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model adalah: {accuracy * 100:.2f}%')

# Simpan model, vectorizer, dan encoder
knn = joblib.load('knn_model_with_city.pkl')
vectorizer = joblib.load('vectorizer_with_city.pkl')
encoder = joblib.load('encoder_with_city.pkl')

def hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis):
    """
    Fungsi ini menghitung skor kecocokan antara keterampilan teknis mahasiswa dan kriteria teknis perusahaan.
    Skor dihitung berdasarkan jumlah keterampilan teknis yang cocok.
    """
    keterampilan_set = set(keterampilan_teknis.lower().split(', '))
    kriteria_set = set(kriteria_teknis.lower().split(', '))

    # Hitung jumlah keterampilan yang cocok
    kecocokan = keterampilan_set.intersection(kriteria_set)
    skor = len(kecocokan) / len(kriteria_set) * 100  # Persentase kecocokan

    return skor, kecocokan
# Fungsi prediksi untuk merekomendasikan perusahaan terbaik
def prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi):
    keterampilan_vec = vectorizer.transform([keterampilan_teknis])
    city_vec = encoder.transform([[preferensi_lokasi]])
    X_new = hstack([keterampilan_vec, city_vec])

    # Ambil 5 tetangga terdekat dan jaraknya
    distances, indices = knn.kneighbors(X_new, n_neighbors=5, return_distance=True)

    prediksi = {}

    for i in range(indices.shape[1]):
        nama_perusahaan = y_train.iloc[indices[0][i]]
        kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]

        # Hitung kecocokan keterampilan teknis dengan kriteria teknis perusahaan
        skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)

        prediksi[nama_perusahaan] = (skor_kecocokan, list(keterampilan_cocok))

        if len(prediksi) == 2:  # Hanya ambil 2 perusahaan terbaik
            break

    return prediksi

# Contoh prediksi
keterampilan_teknis_baru = 'konfigurasi jaringan, penginstalan jaringan, troubleshooting jaringan, html, css, penginstalan server, pengelolaan infrastruktur jaringan, instalasi wi-fi, mikrotik'
preferensi_lokasi_baru = 'padang'
prediksi = prediksi_perusahaan_terbaik(keterampilan_teknis_baru, preferensi_lokasi_baru)

for perusahaan, (skor, keterampilan_cocok) in prediksi.items():
    print(f'Perusahaan: {perusahaan}, Skor kecocokan: {skor:.2f}%, Keterampilan cocok: {", ".join(keterampilan_cocok)}')
