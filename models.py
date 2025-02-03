from db import db  # Mengimpor db dari file db.py

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    keterampilan_teknis = db.Column(db.String(255), nullable=False)
    preferensi_lokasi = db.Column(db.String(255), nullable=False)
    perusahaan = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"<Prediction {self.id}, {self.perusahaan}>"
