import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib

# Sample dataset
# data = {
#     'keterampilan_teknis': [
#         'Penggunaan Excel, verifikasi data, disiplin, Pengembangan Website',
#         'penginstalan Jaringan, Troubleshooting, Monitoring, Manajemen Antivirus, mikrotik',
#         'Penginstalan Server, Pemrograman java, pemrograman C, html, css'
#     ],
#     'kriteria_teknis': [
#         'Penggunaan Excel, verifikasi data, disiplin, Pengembangan Website',
#         'Penginstalan Jaringan, Troubleshooting, Monitoring, Manajemen Antivirus, mikrotik',
#         'Pemrograman java, pemrograman C, html, css'
#     ],
#     'nama_perusahaan': [
#         'Bestpath Network',
#         'PT. APLIKANUSA LINTASARTA',
#         'PT. FIRSTWAP INTERNASIONAL'
#     ],
#     'preferensi_lokasi': ['tangerang', 'padang', 'jakarta selatan'],
#     'company_location': ['tangerang', 'padang', 'jakarta selatan']
# }

df = pd.read_excel('merged_data.xlsx')
df = df.dropna()
# Handle missing values in 'kriteria_teknis'
# df['kriteria_teknis'].fillna('', inplace=True)

# Vectorize the 'kriteria_teknis' column
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df['kriteria_teknis'])

# One-hot encode the company location
encoder = OneHotEncoder()
X_city = encoder.fit_transform(df[['preferensi_lokasi']])

# Combine the text features and the encoded city features
X = hstack([X_text, X_city])

# Use the company names as the target
y = df['nama_perusahaan']

# Train the K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)

# Save the model, vectorizer, and encoder
joblib.dump(knn, 'knn_model_with_city.pkl')
joblib.dump(vectorizer, 'vectorizer_with_city.pkl')
joblib.dump(encoder, 'encoder_with_city.pkl')

# Define a function to predict the best match for keterampilan_teknis and preferred city
