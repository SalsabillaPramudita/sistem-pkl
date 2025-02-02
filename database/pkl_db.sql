CREATE DATABASE pkl_db;

USE pkl_db;

CREATE TABLE mahasiswa (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nama VARCHAR(255),
    program_studi VARCHAR(255),
    keterampilan_teknis TEXT,
    preferensi_lokasi VARCHAR(255)
);

CREATE TABLE perusahaan (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nama_perusahaan VARCHAR(255),
    lokasi_kota VARCHAR(255),
    bidang_industri VARCHAR(255),
    kriteria_teknis TEXT
);
