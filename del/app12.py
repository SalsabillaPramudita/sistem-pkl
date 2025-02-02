from flask import Flask, render_template, request, jsonify
import joblib
from scipy.sparse import hstack
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import nltk

# Pastikan nltk dependency di-download
nltk.download('punkt')

# Kelas khusus untuk stemming
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        stemmer = PorterStemmer()
        return lambda doc: (stemmer.stem(word) for word in analyzer(doc))

# Inisialisasi Flask app
app = Flask(__name__)

# Load model, vectorizer, dan encoder
knn_model = joblib.load('knn_model_with_stemming.pkl')
vectorizer = joblib.load('vectorizer_with_stemming.pkl')
encoder = joblib.load('encoder_with_stemming.pkl')

# Load dataset untuk kriteria teknis
df = pd.read_excel('merged_data.xlsx')

# Fungsi untuk menghitung skor kecocokan
def hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis):
    keterampilan_set = set(keterampilan_teknis.lower().split(', '))
    kriteria_set = set(kriteria_teknis.lower().split(', '))
    kecocokan = keterampilan_set.intersection(kriteria_set)
    skor = len(kecocokan) / len(kriteria_set) * 100 if len(kriteria_set) > 0 else 0
    return skor, kecocokan

# Fungsi untuk memprediksi perusahaan terbaik
def prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi):
    keterampilan_vec = vectorizer.transform([keterampilan_teknis])
    city_vec = encoder.transform([[preferensi_lokasi]])
    X_new = hstack([keterampilan_vec, city_vec])

    distances, indices = knn_model.kneighbors(X_new, n_neighbors=5, return_distance=True)
    prediksi = []

    for i in range(indices.shape[1]):
        nama_perusahaan = df['nama_perusahaan'].iloc[indices[0][i]]
        lokasi_perusahaan = df[df['nama_perusahaan'] == nama_perusahaan]['preferensi_lokasi'].values[0]

        if lokasi_perusahaan.lower() == preferensi_lokasi.lower():
            kriteria_teknis = df[df['nama_perusahaan'] == nama_perusahaan]['kriteria_teknis'].values[0]
            skor_kecocokan, keterampilan_cocok = hitung_skor_kecocokan(keterampilan_teknis, kriteria_teknis)

            prediksi.append({
                'perusahaan': nama_perusahaan,
                'lokasi_perusahaan': lokasi_perusahaan,
                'skor_kecocokan': skor_kecocokan if skor_kecocokan is not None else 'N/A',
                'keterampilan_cocok': keterampilan_cocok
            })

        if len(prediksi) == 2:
            break

    if len(prediksi) == 0:
        prediksi.append({
            'perusahaan': 'Tidak ada perusahaan yang cocok dengan preferensi lokasi.',
            'lokasi_perusahaan': preferensi_lokasi,
            'skor_kecocokan': 'N/A',
            'keterampilan_cocok': []
        })

    return prediksi



# Route untuk halaman prediksi
@app.route('/predict2', methods=['GET'])
def index():
    skills = {
        "Website Development": ["HTML", "CSS", "JavaScript", "PHP", "Laravel", "CodeIgniter", "Bootstrap", "Tailwind"],
        "Mobile Programming": ["Kotlin", "Java", "Flutter", "React Native", "Android Studio", "iOS Development", "Cross-Platform Development"],
        "UI/UX Design": ["Figma", "Adobe XD", "Canva", "Wireframing", "User Interface Design", "User Experience Design", "Design Thinking"],
        "Frontend Development": ["HTML", "CSS", "JavaScript", "Vue.js", "React.js", "Tailwind CSS", "Bootstrap", "Frontend Integration"],
        "Backend Development": ["PHP", "Laravel", "Node.js", "Django", "Express.js", "MySQL", "PostgreSQL", "Spring Boot", "Ruby on Rails", "ASP.NET"],
        "Network Configuration & Troubleshooting": ["Router Configuration", "Switch Configuration", "Mikrotik", "LAN Cable Crimping", "Packet Tracer", "Network Monitoring", "IP Configuration", "Troubleshooting Network Issues", "Access Point Setup"],
        "Database Management": ["MySQL", "PostgreSQL", "CRUD Operations", "Database Optimization", "Data Security", "Relational Databases", "SQL Queries", "XAMPP"],
        "Version Control & Project Collaboration": ["Git", "GitLab", "GitHub", "Version Control", "Project Collaboration", "Team Collaboration"],
        "API Development & Integration": ["RESTful API", "Spring Boot", "Postman", "Frontend-Backend Integration", "CRUD API", "API Testing"],
        "System Installation & Server Management": ["Server Installation", "VM Installation", "Local Network Management", "Server Maintenance", "Operating System Installation"],
        "IT Support & Troubleshooting": ["IT Hardware Troubleshooting", "Software Installation", "Local Network Setup", "CCTV Setup", "Network Issue Troubleshooting", "Infrastructure Maintenance"],
        "Object-Oriented Programming (OOP)": ["Java", "PHP", "Kotlin", "Python", "OOP Principles", "Modular Code", "Reusable Code"],
        "Security & Penetration Testing": ["Kali Linux", "Metasploit", "Vulnerability Identification", "SQL Injection Testing", "Nmap", "Network Security"],
        "Project Management & Collaboration": ["Agile Methodology", "Jira", "Project Planning", "Team Collaboration", "Documentation"],
        "SEO & Web Optimization": ["Core Web Vitals", "Meta Description Optimization", "XML Sitemap", "Schema.org", "SEO Best Practices"],
        "SAP & ERP Systems": ["SAP", "Microsoft Dynamics 365", "Business Process Automation", "Data Management", "SQL Query"]
    }

    return render_template('predict2.html', skills=skills)

@app.route('/predict2', methods=['POST'])
def predict():
    try:
        data = request.json  # Pastikan request data adalah JSON
        keterampilan_teknis = data.get('keterampilan_teknis')
        preferensi_lokasi = data.get('preferensi_lokasi')

        if not keterampilan_teknis or not preferensi_lokasi:
            return jsonify({"error": "Input tidak lengkap"}), 400

        # Proses prediksi dengan model
        perusahaan_prediksi = prediksi_perusahaan_terbaik(keterampilan_teknis, preferensi_lokasi)
        
        return jsonify({
            'keterampilan_teknis': keterampilan_teknis,
            'preferensi_lokasi': preferensi_lokasi,
            'prediksi': perusahaan_prediksi
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
