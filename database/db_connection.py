from flask_sqlalchemy import SQLAlchemy
from flask import Flask

app = Flask(__name__)

# Konfigurasi database menggunakan SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Salsabilla123*@localhost/pkl_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
