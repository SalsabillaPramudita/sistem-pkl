from skills import skills, categorize_skill
from utils import load_model, predict_company

def main():
    print("Selamat datang di sistem prediksi PKL berdasarkan keterampilan teknis.")
    
    # Input data baru
    keterampilan_teknis = input("Masukkan keterampilan teknis Anda (dipisah dengan koma): ")
    preferensi_lokasi = input("Masukkan preferensi lokasi Anda: ")
    
    # Prediksi perusahaan
    prediksi = predict_company(keterampilan_teknis, preferensi_lokasi)
    
    print("\nPrediksi Perusahaan Terbaik Berdasarkan Keterampilan Anda:")
    for perusahaan, (skor, keterampilan_cocok) in prediksi.items():
        print(f"Perusahaan: {perusahaan}, Skor kecocokan: {skor:.2f}%, Keterampilan cocok: {', '.join(keterampilan_cocok)}")

if __name__ == "__main__":
    main()
