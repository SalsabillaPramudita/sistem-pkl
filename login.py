from flask import Blueprint, render_template, request, redirect, url_for, session
from models import Admin  # Import model Admin yang sudah dibuat
from db import db  # Pastikan kamu sudah mengimport db dari db.py

# Definisikan blueprint untuk login
login_bp = Blueprint('login', __name__)

# Route untuk halaman login (GET dan POST)
@login_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('salsabill')
        password = request.form.get('12345678')

        # Cek apakah username dan password cocok dengan data di database
        admin = Admin.query.filter_by(username=username, password=password).first()

        if admin:
            session['admin_id'] = admin.id  # Menyimpan id admin ke dalam session
            return redirect(url_for('dashboard'))  # Redirect ke halaman dashboard setelah login
        else:
            error = 'Username atau password salah'
            return render_template('login.html', error=error)  # Kembali ke halaman login dengan pesan error

    return render_template('login.html')  # Tampilkan halaman login jika metode GET

