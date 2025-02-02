import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_excel('merged_data.xlsx')
df = df.dropna()

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model adalah: {accuracy * 100:.2f}%')

# Save the model, vectorizer, and encoder
joblib.dump(knn, 'knn_model_with_city.pkl')
joblib.dump(vectorizer, 'vectorizer_with_city.pkl')
joblib.dump(encoder, 'encoder_with_city.pkl')

# Define a function to predict the best match for keterampilan_teknis and preferred city
def predict_best_company(keterampilan_teknis, preferensi_lokasi):
    # Vectorize the keterampilan_teknis
    keterampilan_vec = vectorizer.transform([keterampilan_teknis])
    
    # Encode the city
    city_vec = encoder.transform([[preferensi_lokasi]])
    
    # Combine the text and city features
    X_new = hstack([keterampilan_vec, city_vec])
    
    # Predict the best matching company
    prediction = knn.predict(X_new)
    
    return prediction

# Example prediction
keterampilan_teknis_baru = 'Pemrograman java, pemrograman C, html, css'
preferensi_lokasi_baru = 'Jakarta selatan'
prediksi = predict_best_company(keterampilan_teknis_baru, preferensi_lokasi_baru)
print(f'Perusahaan terbaik yang diprediksi: {prediksi[0]}')
