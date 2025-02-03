# db.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db():
    db.create_all()  # Membuat tabel jika belum ada


def init_db(app):
    # Inisialisasi database dengan aplikasi Flask
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/flask_prediksi'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    # Cek koneksi database
    try:
        with app.app_context():
            db.create_all()  # Buat semua tabel yang diperlukan
        print("Koneksi database berhasil!")
    except Exception as e:
        print(f"Error koneksi ke database: {e}")
