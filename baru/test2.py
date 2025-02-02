import joblib
import pandas as pd
from scipy.sparse import hstack

vectorizer = joblib.load('vectorizer_with_city.pkl')
city_encoder = joblib.load('city_encoder_with_city.pkl')
knn = joblib.load('knn_model_with_city.pkl')
label_encoder = joblib.load('label_encoder_with_city.pkl')

df = pd.read_excel('merged_data2.xlsx')
company_names = df['nama_perusahaan'].values  # or df['nama_perusahaan'].tolist()

# Define a function to predict the best match for keterampilan_teknis and preferred city
# def predict_company(skills, user_city):
#     skills_vectorized = vectorizer.transform([skills])
#     city_vectorized = encoder.transform([[user_city]])  # Transform the user's city
#     combined_features = hstack([skills_vectorized, city_vectorized])
    
#     prediction = knn.predict(combined_features)
#     return prediction[0]

# def predict_company(skills, user_city, n_neighbors=3):
#     skills_vectorized = vectorizer.transform([skills])
#     city_vectorized = city_encoder.transform([[user_city]])  # Transform the user's city
#     combined_features = hstack([skills_vectorized, city_vectorized])
    
#     # Get the nearest neighbors
#     distances, indices = knn.kneighbors(combined_features, n_neighbors=n_neighbors)
#     # Retrieve the predicted companies for the nearest neighbors
#     # predicted_companies = knn.predict(combined_features)
#     # unique_predicted_companies = [predicted_companies[i] for i in indices.flatten()]
#     print(distances)
#     predicted_labels = knn._y[indices.flatten()]
#     predicted_companies = label_encoder.inverse_transform(predicted_labels)

#     # Use a set to ensure unique company names, then convert to a list
#     # recommended_companies = list(set(unique_predicted_companies))
    
#     return predicted_companies

def predict_company(skills, user_city, n_neighbors=3):
    skills_vectorized = vectorizer.transform([skills])
    city_vectorized = city_encoder.transform([[user_city]])  # Transform the user's city
    combined_features = hstack([skills_vectorized, city_vectorized])
    
    # Get the nearest neighbors
    distances, indices = knn.kneighbors(combined_features, n_neighbors=n_neighbors)

    # Retrieve the predicted labels and scores
    predicted_labels = knn._y[indices.flatten()]
    predicted_companies = label_encoder.inverse_transform(predicted_labels)

    # Calculate scores based on distances (lower distance = better score)
    scores = 1 / (distances.flatten() + 1e-5)  # Add a small constant to avoid division by zero

    # Combine company names and their scores
    company_scores = {company: score for company, score in zip(predicted_companies, scores)}

    # Sort companies by score in descending order
    sorted_companies = sorted(company_scores.items(), key=lambda x: x[1], reverse=True)

    # Get the best company and its score
    # best_company = sorted_companies[0] if sorted_companies else (None, None)

    return sorted_companies

# Example input
input_skills = "gitlab, basis data, laravel, pengembangan website, object- oriented programming (oop)"
input_city = "tangerang selatan"
predicted_companies = predict_company(input_skills, input_city)

print(f"Recommended company for PKL: {predicted_companies}")
