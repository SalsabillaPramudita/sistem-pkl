import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Sample dataset
# data = {
#     'keterampilan_teknis': [
#         'Penggunaan Excel, verifikasi data, disiplin, Pengembangan Website',
#         'penginstalan Jaringan, Troubleshooting, Monitoring, Manajemen Antivirus, mikrotik',
#         'Penginstalan Server, Pemrograman java, pemograman C, html, css'
#     ],
#     'kriteria_teknis': [
#         'Penggunaan Excel, verifikasi data, disiplin, Pengembangan Website',
#         'Troubleshooting, Monitoring, Manajemen Antivirus, mikrotik',
#         'Pemrograman java, pemrograman C, html, css'
#     ],
#     'nama_perusahaan': [
#         'Bestpath Network',
#         'PT. APLIKANUSA LINTASARTA',
#         'PT. FIRSTWAP INTERNASIONAL'
#     ]
# }

df = pd.read_excel('merged_data.xlsx')
df = df.dropna()
print(df.columns)

# Vectorize the 'keterampilan_teknis' and 'kriteria_teknis'
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['kriteria_teknis'])

# Use the company names as the target
y = df['nama_perusahaan']

# Train the K-Nearest Neighbors model on the entire dataset
knn = KNeighborsClassifier(n_neighbors=1)  # Set to 1 due to small dataset
knn.fit(X, y)

joblib.dump(knn, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# # Define a function to predict the best match for keterampilan_teknis and return the company name
# def predict_company(skills):
#     skills_vectorized = vectorizer.transform([skills])
#     prediction = knn.predict(skills_vectorized)
#     print(prediction)
#     return prediction[0]

# # Example input
# input_skills = "Penggunaan Excel, disiplin, Pengembangan Website, penginstalan Jaringan, Troubleshooting, Monitoring, Manajemen Antivirus, mikrotik"
# predicted_company = predict_company(input_skills)
# print(f"Recommended company for PKL: {predicted_company}")
