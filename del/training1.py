import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load dataset
df = pd.read_excel('merged_data.xlsx')
df = df.dropna()

# Vectorize the 'kriteria_teknis' column
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df['kriteria_teknis'])

# One-hot encode the company location with updated argument
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_city = encoder.fit_transform(df[['preferensi_lokasi']])

# Combine the text features and the encoded city features
X = hstack([X_text, X_city])

# Use the company names as the target
y = df['nama_perusahaan']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model adalah: {accuracy * 100:.2f}%')

# Save the model, vectorizer, and encoder
joblib.dump(knn, 'models/knn_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump(encoder, 'models/encoder.pkl')

# Predicting new input
def predict_company(keterampilan_teknis, preferensi_lokasi):
    keterampilan_vec = vectorizer.transform([keterampilan_teknis])
    city_vec = encoder.transform([[preferensi_lokasi]])
    X_new = hstack([keterampilan_vec, city_vec])

    distances, indices = knn.kneighbors(X_new, n_neighbors=5, return_distance=True)

    # Handling potential index errors
    prediksi = []
    for i in range(len(indices[0])):
        try:
            nama_perusahaan = y_train.iloc[indices[0][i]]
            prediksi.append(nama_perusahaan)
        except IndexError as e:
            print(f"Error accessing index {indices[0][i]}: {str(e)}")
            continue

    return prediksi

# Test prediction
keterampilan_teknis = "HTML, CSS, Laravel"
preferensi_lokasi = "bandung"
prediksi = predict_company(keterampilan_teknis, preferensi_lokasi)
print("Prediksi perusahaan:", prediksi)
