# from flask import Flask, render_template, request
# import joblib
# from scipy.sparse import hstack
# from sklearn.metrics import accuracy_score
# app = Flask(__name__)

# # Muat model, vectorizer, dan encoder
# vectorizer = joblib.load('vectorizer_with_city.pkl')
# city_encoder = joblib.load('city_encoder_with_city.pkl')
# knn = joblib.load('knn_model_with_city.pkl')
# label_encoder = joblib.load('label_encoder_with_city.pkl')

# def prediksi_perusahaan(keterampilan_teknis, preferensi_lokasi, n_neighbors=3):
#     keterampilan_vec = vectorizer.transform([keterampilan_teknis])
#     kota_vec = city_encoder.transform([[preferensi_lokasi]])  # Vektorisasi preferensi lokasi
#     fitur_gabungan = hstack([keterampilan_vec, kota_vec])

#     # Dapatkan tetangga terdekat dan jaraknya
#     distances, indices = knn.kneighbors(fitur_gabungan, n_neighbors=n_neighbors)

#     # Ambil label dan skor
#     predicted_labels = knn._y[indices.flatten()]
#     predicted_companies = label_encoder.inverse_transform(predicted_labels)

#     # Hitung skor berdasarkan jarak
#     scores = 1 / (distances.flatten() + 1e-5)

#     # Gabungkan nama perusahaan dengan skornya
#     perusahaan_skors = {company: score for company, score in zip(predicted_companies, scores)}

#     # Urutkan berdasarkan skor tertinggi
#     perusahaan_terurut = sorted(perusahaan_skors.items(), key=lambda x: x[1], reverse=True)

#     return perusahaan_terurut

# accuracy = accuracy_score(knn._y, )
# print(f'Akurasi model adalah: {accuracy * 100:.2f}%')

# @app.route('/')
# def landing():
#     return render_template('landing.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/predict')
# def index():
#     return render_template('predict3.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Ambil input dari form HTML
#     keterampilan_teknis = request.form.get('keterampilan_teknis')
#     preferensi_lokasi = request.form.get('preferensi_lokasi')

#     # Prediksi perusahaan
#     hasil_prediksi = prediksi_perusahaan(keterampilan_teknis, preferensi_lokasi)

#     return render_template('hasil_prediksi.html', hasil_prediksi=hasil_prediksi)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import joblib
from scipy.sparse import hstack
import skills  # Mengimpor skills.py untuk keterampilan teknis

app = Flask(__name__)

# Muat model, vectorizer, dan encoder
vectorizer = joblib.load('vectorizer_with_city.pkl')
city_encoder = joblib.load('city_encoder_with_city.pkl')
knn = joblib.load('knn_model_with_city.pkl')
label_encoder = joblib.load('label_encoder_with_city.pkl')

# Fungsi untuk prediksi perusahaan berdasarkan keterampilan teknis dan preferensi lokasi
def prediksi_perusahaan(keterampilan_teknis, preferensi_lokasi, n_neighbors=3):
    keterampilan_vec = vectorizer.transform([keterampilan_teknis])
    
    # Cek apakah preferensi lokasi ada dalam data pelatihan, jika tidak, beri fallback
    try:
        kota_vec = city_encoder.transform([[preferensi_lokasi]])
    except ValueError:
        print(f"Lokasi {preferensi_lokasi} tidak ditemukan dalam data pelatihan, menggunakan fallback.")
        kota_vec = city_encoder.transform([['unknown_city']])  # Atur fallback ke kota default
    
    fitur_gabungan = hstack([keterampilan_vec, kota_vec])

    # Dapatkan tetangga terdekat dan jaraknya
    distances, indices = knn.kneighbors(fitur_gabungan, n_neighbors=n_neighbors)

    # Ambil label dan skor (jarak)
    predicted_labels = knn._y[indices.flatten()]
    predicted_companies = label_encoder.inverse_transform(predicted_labels)

    # Gabungkan nama perusahaan dengan jaraknya
    perusahaan_terurut = [(company, distance) for company, distance in zip(predicted_companies, distances.flatten())]

    # Urutkan berdasarkan jarak terdekat
    perusahaan_terurut = sorted(perusahaan_terurut, key=lambda x: x[2])

    return perusahaan_terurut

# Routing halaman utama
@app.route('/')
def landing():
    return render_template('landing.html')

# Routing halaman about
@app.route('/about')
def about():
    return render_template('about.html')

# Routing halaman predict (GET untuk menampilkan form)
@app.route('/predict')
def index():
    # Mengambil skill list dari skills.py
    skill_categories = skills.skill_list
    return render_template('predict3.html', skill_categories=skill_categories)

# Routing halaman predict (POST untuk hasil prediksi)
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari form HTML
    keterampilan_teknis = request.form.getlist('keterampilan_teknis')  # Mendapatkan keterampilan sebagai list
    preferensi_lokasi = request.form.get('preferensi_lokasi')

    if not keterampilan_teknis or not preferensi_lokasi:
        # Jika input tidak lengkap, tampilkan pesan error di halaman prediksi
        error = "Pastikan untuk mengisi keterampilan teknis dan preferensi lokasi."
        return render_template('predict3.html', error=error, skill_categories=skills.skill_list)

    # Gabungkan keterampilan teknis menjadi satu string
    keterampilan_teknis_str = ', '.join(keterampilan_teknis)

    # Prediksi perusahaan
    hasil_prediksi = prediksi_perusahaan(keterampilan_teknis_str, preferensi_lokasi)

    return render_template('hasil_prediksi.html', hasil_prediksi=hasil_prediksi)

if __name__ == '__main__':
    app.run(debug=True)
