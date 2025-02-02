# import joblib
# import pandas as pd
# from scipy.sparse import hstack
# from sklearn.metrics import accuracy_score

# vectorizer = joblib.load('vectorizer_with_city.pkl')
# city_encoder = joblib.load('city_encoder_with_city.pkl')
# knn = joblib.load('knn_model_with_city.pkl')
# label_encoder = joblib.load('label_encoder_with_city.pkl')

# df = pd.read_excel('merged_data.xlsx')
# company_names = df['nama_perusahaan'].values  # or df['nama_perusahaan'].tolist()

# # Define a function to predict the best match for keterampilan_teknis and preferred city
# # def predict_company(skills, user_city):
# #     skills_vectorized = vectorizer.transform([skills])
# #     city_vectorized = encoder.transform([[user_city]])  # Transform the user's city
# #     combined_features = hstack([skills_vectorized, city_vectorized])
    
# #     prediction = knn.predict(combined_features)
# #     return prediction[0]

# # def predict_company(skills, user_city, n_neighbors=3):
# #     skills_vectorized = vectorizer.transform([skills])
# #     city_vectorized = city_encoder.transform([[user_city]])  # Transform the user's city
# #     combined_features = hstack([skills_vectorized, city_vectorized])
    
# #     # Get the nearest neighbors
# #     distances, indices = knn.kneighbors(combined_features, n_neighbors=n_neighbors)
# #     # Retrieve the predicted companies for the nearest neighbors
# #     # predicted_companies = knn.predict(combined_features)
# #     # unique_predicted_companies = [predicted_companies[i] for i in indices.flatten()]
# #     print(distances)
# #     predicted_labels = knn._y[indices.flatten()]
# #     predicted_companies = label_encoder.inverse_transform(predicted_labels)

# #     # Use a set to ensure unique company names, then convert to a list
# #     # recommended_companies = list(set(unique_predicted_companies))
    
# #     return predicted_companies

# def predict_company(skills, user_city, n_neighbors=3):
#     skills_vectorized = vectorizer.transform([skills])
#     city_vectorized = city_encoder.transform([[user_city]])  # Transform the user's city
#     combined_features = hstack([skills_vectorized, city_vectorized])
    
#     # Get the nearest neighbors
#     distances, indices = knn.kneighbors(combined_features, n_neighbors=n_neighbors)

#     # Retrieve the predicted labels and scores
#     predicted_labels = knn._y[indices.flatten()]
#     predicted_companies = label_encoder.inverse_transform(predicted_labels)

#     # Calculate scores based on distances (lower distance = better score)
#     scores = 1 / (distances.flatten() + 1e-5)  # Add a small constant to avoid division by zero

#     # Combine company names and their scores
#     company_scores = {company: score for company, score in zip(predicted_companies, scores)}

#     # Sort companies by score in descending order
#     sorted_companies = sorted(company_scores.items(), key=lambda x: x[1], reverse=True)

#     # Get the best company and its score
#     # best_company = sorted_companies[0] if sorted_companies else (None, None)

#     return sorted_companies

# #menghitung akurasi
# # Menghitung akurasi
# accuracy = accuracy_score(knn._y, predict_company)
# print(f'Akurasi model adalah: {accuracy * 100:.2f}%')



# # Example input
# input_skills = "gitlab, basis data, laravel, pengembangan website, object- oriented programming (oop)"
# input_city = "tangerang selatan"
# predicted_companies = predict_company(input_skills, input_city)

# print(f"Recommended company for PKL: {predicted_companies}")
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import numpy as np
import warnings

# Load model, vectorizer, dan encoder
vectorizer = joblib.load('vectorizer_with_city.pkl')
city_encoder = joblib.load('city_encoder_with_city.pkl')
knn = joblib.load('knn_model_with_city.pkl')
label_encoder = joblib.load('label_encoder_with_city.pkl')

# Load dataset
df = pd.read_excel('merged_data.xlsx')
warnings.filterwarnings("ignore", category=UserWarning)

# Fungsi untuk prediksi
def predict_company(skills, user_city, n_neighbors=3):
    skills_vectorized = vectorizer.transform([skills])
    try:
        city_vectorized = city_encoder.transform([[user_city]])
    except ValueError:
        # print(f"Lokasi {user_city} tidak ditemukan dalam data pelatihan, menggunakan fallback.")
        city_vectorized = np.zeros((1, city_encoder.categories_[0].shape[0]))

    combined_features = hstack([skills_vectorized, city_vectorized])
    distances, indices = knn.kneighbors(combined_features, n_neighbors=n_neighbors)
    predicted_labels = knn._y[indices.flatten()]
    predicted_companies = label_encoder.inverse_transform(predicted_labels)

    return predicted_companies[0]

# Fungsi untuk menghitung akurasi
def calculate_accuracy(X, y):
    predicted_labels = []
    for i in range(len(X)):
        skills, city = X[i]
        sorted_companies = predict_company(skills, city)
        predicted_labels.append(sorted_companies)
    
    return accuracy_score(y, predicted_labels)

# Persiapan data: gabungkan keterampilan dan preferensi lokasi
X = list(zip(df['kriteria_teknis'].values, df['preferensi_lokasi'].values))
y = df['nama_perusahaan'].values

# Pisahkan data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.metrics import classification_report, confusion_matrix

# Prediksi pada data uji
y_pred_test = [predict_company(skills, city) for skills, city in X_test]

# Hasil laporan klasifikasi
print(classification_report(y_test, y_pred_test))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred_test))

# Hitung akurasi pada data training
train_accuracy = calculate_accuracy(X_train, y_train)
print(f'Akurasi pada data training: {train_accuracy * 100:.2f}%')

# Hitung akurasi pada data testing
test_accuracy = calculate_accuracy(X_test, y_test)
print(f'Akurasi pada data testing: {test_accuracy * 100:.2f}%')