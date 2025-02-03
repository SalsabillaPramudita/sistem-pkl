# app.py

from flask import Flask, render_template, request, redirect, url_for
import joblib
from scipy.sparse import hstack
import skills  # Mengimpor skills.py untuk keterampilan teknis
from db import db, init_db  # Mengimpor objek db dan fungsi init_db dari db.py
from models import Prediction  # Model Prediction yang digunakan untuk menyimpan data


app = Flask(__name__)

# Konfigurasi koneksi ke MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/flask_prediksi'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)  # Inisialisasi database
with app.app_context():
    db.create_all()  # Membuat tabel yang didefinisikan di model


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
    perusahaan_terurut = sorted(perusahaan_terurut, key=lambda x: x[1])  # x[1] adalah distance

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

    return render_template('hasil_prediksi.html', 
                           hasil_prediksi=hasil_prediksi, 
                           keterampilan_teknis=keterampilan_teknis_str, 
                           preferensi_lokasi=preferensi_lokasi)


@app.route('/save_prediction', methods=['POST'])
def save_prediction():
    keterampilan_teknis = request.form.get('keterampilan_teknis')
    preferensi_lokasi = request.form.get('preferensi_lokasi')
    perusahaan = request.form.get('perusahaan')

    if keterampilan_teknis and preferensi_lokasi and perusahaan:
        try:
            new_prediction = Prediction(
                keterampilan_teknis=keterampilan_teknis,
                preferensi_lokasi=preferensi_lokasi,
                perusahaan=perusahaan
            )
            db.session.add(new_prediction)
            db.session.commit()
            return "Prediksi berhasil disimpan."
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return "Gagal menyimpan prediksi."
    else:
        return "Input tidak lengkap."

if __name__ == '__main__':
    app.run(debug=True)
