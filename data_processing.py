import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Caching Stemmer untuk mempercepat proses
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stem_cache = {}

# Fungsi untuk melakukan preprocessing teks, seperti stemming dan tokenisasi
def preprocess_text(text):
    """ Fungsi untuk preprocessing teks: stemming dan tokenisasi """
    if isinstance(text, float):  # Jika tipe data float, kembalikan string kosong
        return []
    
    try:
        text = text.lower()  # Konversi teks menjadi huruf kecil
        tokens = text.split(', ')  # Pisahkan berdasarkan koma
        processed_tokens = []
        for token in tokens:
            if token in stem_cache:
                processed_tokens.append(stem_cache[token])
            else:
                stemmed_token = stemmer.stem(token)  # Lakukan stemming
                stem_cache[token] = stemmed_token
                processed_tokens.append(stemmed_token)
        return processed_tokens
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return []  # Kembalikan list kosong jika terjadi error

# Fungsi untuk menghitung skor kecocokan antara keterampilan mahasiswa dan kriteria teknis perusahaan
def hitung_skor_kecocokan(keterampilan_mahasiswa, kriteria_teknis):
    keterampilan_mahasiswa = set(preprocess_text(keterampilan_mahasiswa))
    kriteria_teknis = set(preprocess_text(kriteria_teknis))
    kecocokan = keterampilan_mahasiswa.intersection(kriteria_teknis)  # Cek keterampilan yang cocok
    return len(kecocokan)  # Mengembalikan jumlah keterampilan yang cocok

# Fungsi untuk one-hot encoding fitur kategorikal seperti program studi dan lokasi
def encode_categorical(data, columns, encoder=None):
    if encoder is None:  # Jika tidak ada encoder, buat yang baru
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(data[columns])  # Lakukan one-hot encoding
    else:
        encoded_data = encoder.transform(data[columns])  # Gunakan encoder yang sudah dilatih
    
    return pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out()), encoder  # Return sebagai DataFrame dan encoder

# Fungsi untuk preprocessing data mahasiswa dan perusahaan serta menghitung skor kecocokan
def preprocess_data(data_mahasiswa, data_perusahaan, encoder=None):
    # Pastikan tidak ada nilai NaN pada kolom keterampilan teknis dan kriteria teknis
    data_mahasiswa['keterampilan_teknis'] = data_mahasiswa['keterampilan_teknis'].fillna('')
    data_perusahaan['kriteria_teknis'] = data_perusahaan['kriteria_teknis'].fillna('')

    # Encoding program studi dan preferensi lokasi
    encoded_program_studi, encoder = encode_categorical(data_mahasiswa, ['program_studi'], encoder)
    encoded_preferensi_lokasi, encoder = encode_categorical(data_mahasiswa, ['preferensi_lokasi'], encoder)
    
    # Menggabungkan hasil encoding dengan data mahasiswa
    data_mahasiswa_encoded = pd.concat([data_mahasiswa.reset_index(drop=True), encoded_program_studi, encoded_preferensi_lokasi], axis=1)

    # Proses matching mahasiswa dengan perusahaan dan hitung skor kecocokan
    hasil_preprocessing = []
    for idx_mhs, mahasiswa in data_mahasiswa_encoded.iterrows():
        for idx_perusahaan, perusahaan in data_perusahaan.iterrows():
            skor_kecocokan = hitung_skor_kecocokan(mahasiswa['keterampilan_teknis'], perusahaan['kriteria_teknis'])
            hasil_preprocessing.append({
                'mahasiswa': mahasiswa['nama'],
                'perusahaan': perusahaan['nama_perusahaan'],
                'skor_kecocokan': skor_kecocokan,
                'program_studi': mahasiswa['program_studi'],
                'preferensi_lokasi': mahasiswa['preferensi_lokasi']
            })
    
    return pd.DataFrame(hasil_preprocessing), encoder  # Return DataFrame hasil preprocessing dan encoder

# Fungsi khusus untuk prediksi data baru
def preprocess_data_prediction(data_mahasiswa_encoded, data_perusahaan):
    """ Fungsi ini digunakan untuk melakukan preprocessing pada data prediksi """
    hasil_preprocessing = []
    for idx_mhs, mahasiswa in data_mahasiswa_encoded.iterrows():
        for idx_perusahaan, perusahaan in data_perusahaan.iterrows():
            skor_kecocokan = hitung_skor_kecocokan(mahasiswa['keterampilan_teknis'], perusahaan['kriteria_teknis'])
            hasil_preprocessing.append({
                'mahasiswa': mahasiswa.get('nama', 'Unknown'),
                'perusahaan': perusahaan['nama_perusahaan'],
                'skor_kecocokan': skor_kecocokan,
                'program_studi': mahasiswa['program_studi'],
                'preferensi_lokasi': mahasiswa['preferensi_lokasi']
            })
    
    return pd.DataFrame(hasil_preprocessing)

# Fungsi untuk menyimpan hasil preprocessing ke dalam file jika diperlukan
def save_preprocessed_data(data, filepath):
    """ Fungsi ini digunakan untuk menyimpan hasil preprocessing """
    try:
        data.to_excel(filepath, index=False)
        print(f"Hasil preprocessing disimpan di: {filepath}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")

