from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
import joblib
from scipy.sparse import hstack
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Definisikan ulang StemmedTfidfVectorizer di sini
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        stemmer = PorterStemmer()
        return lambda doc: (stemmer.stem(word) for word in analyzer(doc))

# Inisialisasi Flask app
app = Flask(__name__)

# Load model, vectorizer, dan encoder setelah definisi class
knn_model = joblib.load('knn_model_with_stemming.pkl')
vectorizer = joblib.load('vectorizer_with_stemming.pkl')
encoder = joblib.load('encoder_with_stemming.pkl')

# Memuat ulang dataset untuk mengambil y_train
df = pd.read_excel(r'D:/sistem prediksi pkl/backend/merged_data.xlsx')
df = df.dropna()
y_train = df['nama_perusahaan']

def hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis):
    keterampilan_set = set(keterampilan_teknis.lower().split(', '))
    kriteria_set = set(kriteria_teknis.lower().split(', '))
    kecocokan = keterampilan_set.intersection(kriteria_set)
    skor = len(kecocokan) / len(kriteria_set) * 100 if len(kriteria_set) > 0 else 0
    return skor, kecocokan

# Fungsi prediksi untuk merekomendasikan perusahaan terbaik
def predict_best_company(keterampilan_teknis, preferensi_lokasi):
    keterampilan_vec = vectorizer.transform([keterampilan_teknis])
    city_vec = encoder.transform([[preferensi_lokasi]])
    X_new = hstack([keterampilan_vec, city_vec])

    # Ambil 5 tetangga terdekat dan jaraknya
    distances, indices = knn_model.kneighbors(X_new, n_neighbors=5, return_distance=True)

    prediksi = {}
    for i in range(indices.shape[1]):
        nama_perusahaan = y_train.iloc[indices[0][i]]
        kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]

        # Hitung kecocokan keterampilan teknis dengan kriteria teknis perusahaan
        skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)
        prediksi[nama_perusahaan] = (skor_kecocokan, list(keterampilan_cocok))

        if len(prediksi) == 2:
            break

    return prediksi

@app.route('/')
def index():
    return render_template('predict3.html')

@app.route('/predict3', methods=['POST'])
def predict():
    try:
        # Ambil data dari request
        keterampilan_teknis = request.form.get('keterampilan_teknis')
        preferensi_lokasi = request.form.get('preferensi_lokasi')

        # Lakukan prediksi
        prediksi = predict_best_company(keterampilan_teknis, preferensi_lokasi)
        return jsonify(prediksi)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
