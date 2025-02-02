from flask import Flask, request, jsonify, render_template
import joblib
from scipy.sparse import hstack


# Load the saved model, vectorizer, and encoder
knn = joblib.load('knn_model_with_city.pkl')
vectorizer = joblib.load('vectorizer_with_city.pkl')
encoder = joblib.load('encoder_with_city.pkl')

app = Flask(__name__)

def hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis):
    keterampilan_set = set(keterampilan_teknis.lower().split(', '))
    kriteria_set = set(kriteria_teknis.lower().split(', '))
    kecocokan = keterampilan_set.intersection(kriteria_set)
    skor = len(kecocokan) / len(kriteria_set) * 100
    return skor, kecocokan

@app.route('/')
def landing_page():
    return render_template('landing.html')  # Correct spelling
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Serve a predict page or return some default response for GET request
        return render_template('predict.html')  # Make sure predict.html exists in templates folder

    # POST method logic (your existing prediction logic)
    data = request.get_json()
    keterampilan_teknis = data['keterampilan_teknis']
    preferensi_lokasi = data['preferensi_lokasi']
    keterampilan_vec = vectorizer.transform([keterampilan_teknis])
    city_vec = encoder.transform([[preferensi_lokasi]])
    X_new = hstack([keterampilan_vec, city_vec])
    distances, indices = knn.kneighbors(X_new, n_neighbors=5, return_distance=True)

    prediksi = {}
    for i in range(indices.shape[1]):
        nama_perusahaan = df['nama_perusahaan'].iloc[indices[0][i]]
        kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]
        skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)
        prediksi[nama_perusahaan] = {
            'skor_kecocokan': skor_kecocokan,
            'keterampilan_cocok': list(keterampilan_cocok)
        }
        if len(prediksi) == 2:
            break

    return jsonify(prediksi)
if __name__ == '__main__':
    app.run(debug=True)
