from flask import Flask, render_template, request, jsonify
import joblib
from scipy.sparse import hstack
import pandas as pd

# Inisialisasi Flask app
app = Flask(__name__)

# Load model, vectorizer, dan encoder
knn_model = joblib.load('knn_model_with_city.pkl')
vectorizer = joblib.load('vectorizer_with_city.pkl')
encoder = joblib.load('encoder_with_city.pkl')

# Load dataset for reference to match skills and companies
df = pd.read_excel('merged_data.xlsx')
df = df.dropna()

# Function to calculate skill matching score
# def hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis):
#     keterampilan_set = set(keterampilan_teknis.lower().split(', '))
#     kriteria_set = set(kriteria_teknis.lower().split(', '))

#     # Hitung jumlah keterampilan yang cocok
#     kecocokan = keterampilan_set.intersection(kriteria_set)
#     skor = len(kecocokan) / len(kriteria_set) * 100  # Persentase kecocokan

#     return skor, kecocokan
def hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis):
    keterampilan_set = set(keterampilan_teknis.lower().split(', '))
    kriteria_set = set(kriteria_teknis.lower().split(', '))
    kecocokan = keterampilan_set.intersection(kriteria_set)
    skor = len(kecocokan) / len(kriteria_set) * 100
    return skor, kecocokan

# Prediction function for the best company recommendation
# def prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi):
#     keterampilan_vec = vectorizer.transform([keterampilan_teknis])
#     city_vec = encoder.transform([[preferensi_lokasi]])
#     X_new = hstack([keterampilan_vec, city_vec])

#     # Ambil 5 tetangga terdekat dan jaraknya
#     distances, indices = knn_model.kneighbors(X_new, n_neighbors=5, return_distance=True)

#     prediksi = {}
#     print(df)

#     for i in range(indices.shape[1]):
#         nama_perusahaan = df.iloc[indices[0][i]]['nama_perusahaan']
#         kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]
#         # kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]
#         # Hitung kecocokan keterampilan teknis dengan kriteria teknis perusahaan
#         skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)

#         prediksi[nama_perusahaan] = (skor_kecocokan, list(keterampilan_cocok))

#         # if len(prediksi) == 2:  # Hanya ambil 2 perusahaan terbaik
#         #     break

#     return prediksi

def prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi):
    keterampilan_vec = vectorizer.transform([keterampilan_teknis])
    city_vec = encoder.transform([[preferensi_lokasi]])
    X_new = hstack([keterampilan_vec, city_vec])

    distances, indices = knn_model.kneighbors(X_new, n_neighbors=5, return_distance=True)
    
    prediksi = {}
    for i in range(indices.shape[1]):
        nama_perusahaan = df.iloc[indices[0][i]]['nama_perusahaan']
        lokasi_perusahaan = df.iloc[indices[0][i]]['preferensi_lokasi']
        
        if lokasi_perusahaan.lower() == preferensi_lokasi.lower():
            kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]
            skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)
            prediksi[nama_perusahaan] = (skor_kecocokan, list(keterampilan_cocok))
    
    return prediksi
# def prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi):
#     # Transform input skills and location
#     keterampilan_vec = vectorizer.transform([keterampilan_teknis])
#     city_vec = encoder.transform([[preferensi_lokasi]])
    
#     # Combine text features and location features
#     X_new = hstack([keterampilan_vec, city_vec])

#     print("X NEW: ",X_new)
#     print("X_new shape: ", X_new.shape)


#     # Get the nearest neighbors and distances
#     distances, indices = knn_model.kneighbors(X_new, n_neighbors=3, return_distance=True)
#     print(indices)
#     p = knn_model.predict(X_new)
#     print(df)
#     print(p)
#     prediksi = {}

#     for i in range(indices.shape[1]):
#         # Gunakan df untuk mendapatkan nama_perusahaan berdasarkan index hasil prediksi
#         nama_perusahaan = df.iloc[indices[0][i]]['nama_perusahaan']
#         # kriteria_teknis = df.iloc[indices[0][i]]['kriteria_teknis']
#         kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]

#         # Calculate the skill match score between user skills and company criteria
#         skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)

#         prediksi[nama_perusahaan] = {
#             'skor_kecocokan': int(skor_kecocokan),
#             'keterampilan_cocok': list(keterampilan_cocok)
#         }

#         print(prediksi)

#         # if len(prediksi) == 2:  # Limit to 2 best matches
#         #     break

#     prediksi = sorted(prediksi, key=lambda x: x['skor_kecocokan'], reverse=True)

#     return prediksi


# Route untuk halaman prediksi (GET) - menampilkan form
# @app.route('/predict', methods=['GET'])
# def index():
#     return render_template('predict.html')

# # Route untuk melakukan prediksi berdasarkan input pengguna (POST)
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Ambil data dari request form (input dari HTML form)
#         keterampilan_teknis = request.form.get('keterampilan_teknis')
#         preferensi_lokasi = request.form.get('preferensi_lokasi')

#         if not keterampilan_teknis or not preferensi_lokasi:
#             return render_template('predict.html', error="Input tidak lengkap")

#         # Lakukan prediksi
#         perusahaan_prediksi = prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi)

#         # Tampilkan hasil prediksi di halaman HTML
#         return render_template('predict.html', prediksi=perusahaan_prediksi)

#     except Exception as e:
#         print(f"Error terjadi: {e}")
#         return render_template('predict.html', error="Terjadi kesalahan saat melakukan prediksi")

# # Route untuk halaman landing
# @app.route('/')
# def landing():
#     return render_template('landing.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# # Jalankan aplikasi
# if __name__ == '__main__':
#     app.run(debug=True)
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         nama = request.form['nama']
#         keterampilan_teknis = request.form['keterampilan_teknis']
#         preferensi_lokasi = request.form['preferensi_lokasi']
        
        
#         # Lakukan prediksi
#         hasil_prediksi = prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi)

#         # Kembalikan hasil prediksi ke template
#         return render_template('predict.html', nama=nama, hasil_prediksi=hasil_prediksi)

#     return render_template('predict.html')

# # Menjalankan aplikasi Flask
# if __name__ == '__main__':
#     app.run(debug=True)
# #
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        nama = request.form['nama']
        keterampilan_teknis = request.form['keterampilan_teknis']
        preferensi_lokasi = request.form['preferensi_lokasi']

        # Debugging: Print input data to check if it's correctly received
        print("Nama:", nama)
        print("Keterampilan Teknis:", keterampilan_teknis)
        print("Preferensi Lokasi:", preferensi_lokasi)

        # Prediksi tempat PKL berdasarkan input pengguna
        try:
            prediksi = prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi)

            # Debugging: Print prediction result
            print("Hasil Prediksi:", prediksi)

            return render_template('predict.html', prediksi=prediksi, nama=nama)
        except Exception as e:
            print("Error:", str(e))
            return render_template('predict.html', error=str(e))
        
if __name__ == '__main__':
    app.run(debug=True)
#