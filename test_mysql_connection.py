import pymysql

# Ganti dengan kredensial Anda
host = 'localhost'  # Alamat server MySQL
user = 'root'       # Username MySQL
password = ''       # Password MySQL (kosongkan jika tidak ada password)
database = 'flask_prediksi'  # Nama database yang ingin Anda hubungkan

try:
    # Mencoba untuk melakukan koneksi ke database MySQL
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=database
    )
    print("Koneksi ke MySQL berhasil!")
    
    # Anda bisa mencoba untuk menjalankan query di sini
    with connection.cursor() as cursor:
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        print(f"Versi MySQL: {version[0]}")

    # Jangan lupa untuk menutup koneksi setelah selesai
    connection.close()

except pymysql.MySQLError as e:
    print(f"Terjadi kesalahan saat mencoba koneksi ke MySQL: {e}")
