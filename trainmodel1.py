# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder
# from scipy.sparse import hstack
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib

# # Load dataset
# df = pd.read_excel('merged_data.xlsx')
# df = df.dropna()

# # Vectorize the 'kriteria_teknis' column (Text data)
# vectorizer = TfidfVectorizer()
# X_kriteria = vectorizer.fit_transform(df['kriteria_teknis'])

# # One-hot encode the 'preferensi_lokasi' column (Categorical data)
# encoder = OneHotEncoder()
# X_lokasi = encoder.fit_transform(df[['preferensi_lokasi']])

# # Combine both 'kriteria_teknis' and 'preferensi_lokasi' features into one feature matrix
# X = hstack([X_kriteria, X_lokasi])

# # Use the company names as the target variable
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

# # Function to calculate skill matching score
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

# # Prediction function for the best company recommendation
# def prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi):
#     # Transform input skills and location
#     keterampilan_vec = vectorizer.transform([keterampilan_teknis])
#     lokasi_vec = encoder.transform([[preferensi_lokasi]])
    
#     # Debugging prints (optional)
#     print(f"Encoded keterampilan_teknis: {keterampilan_vec}")
#     print(f"Encoded preferensi_lokasi: {lokasi_vec}")
    
#     # Combine the processed input features into one feature matrix
#     X_new = hstack([keterampilan_vec, lokasi_vec])

#     # Debugging prints (optional)
#     print(f"Combined X_new: {X_new}")
    
#     # Get the nearest neighbors and distances
#     distances, indices = knn.kneighbors(X_new, n_neighbors=3, return_distance=True)

#     prediksi = {}

#     for i in range(indices.shape[1]):
#         nama_perusahaan = y_train.iloc[indices[0][i]]
#         kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]

#         # Calculate the skill match score between user skills and company criteria
#         skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)

#         prediksi[nama_perusahaan] = {
#             'skor_kecocokan': skor_kecocokan,
#             'keterampilan_cocok': list(keterampilan_cocok)
#         }

#         if len(prediksi) == 2:  # Limit to 2 best matches
#             break

#     return prediksi

# # Example of using the prediction function
# keterampilan_teknis_baru = 'laravel, Desain UI/UX, mysql, pengolahan data, Pengembangan backend, Pengembangan Frontend, basis data, sistem versi kontrol, html, css, php, javascript'
# preferensi_lokasi_baru = 'jakarta selatan'

# # Make the prediction
# prediksi = prediksi_perusahaan_terbaik(keterampilan_teknis_baru, preferensi_lokasi_baru)

# # Output formatted predictions
# for perusahaan, data in prediksi.items():
#     print(f'Perusahaan: {perusahaan}')
#     print(f'Skor Kecocokan: {data["skor_kecocokan"]:.2f}%')
#     print(f'Keterampilan yang Cocok: {", ".join(data["keterampilan_cocok"])}')
#     print('--------------------------------------------')
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
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Prediksi dengan data uji
y_pred = knn.predict(X)

# Menghitung akurasi
accuracy = accuracy_score(X, y, y_pred)
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
    p = knn.predict(X_new)
    print(p)
    print(indices)
    print(df)
    for i in range(indices.shape[1]):
        nama_perusahaan = df.iloc[indices[0][i]]['nama_perusahaan']
        kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]

        # Hitung kecocokan keterampilan teknis dengan kriteria teknis perusahaan
        skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)

        prediksi[nama_perusahaan] = (skor_kecocokan, list(keterampilan_cocok))

        # if len(prediksi) == 2:  # Hanya ambil 2 perusahaan terbaik
        #     break

    return prediksi

# Contoh prediksi
keterampilan_teknis_baru = 'pengembangan website, basis data, pemograman mobile, desain ui/ux, Excel, penggunaan SAP, Komunikasi efektif, bekerja dalam tim, Penginstalan Jaringan, konfigurasi jaringan'
preferensi_lokasi_baru = 'jakarta selatan'
prediksi = prediksi_perusahaan_terbaik(keterampilan_teknis_baru, preferensi_lokasi_baru)
for perusahaan, (skor, keterampilan_cocok) in prediksi.items():
    print(f'Perusahaan: {perusahaan}, Skor kecocokan: {skor:.2f}%, Keterampilan cocok: {", ".join(keterampilan_cocok)}')