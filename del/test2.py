import joblib
from scipy.sparse import hstack

vectorizer = joblib.load('vectorizer_with_city.pkl')
encoder = joblib.load('encoder_with_city.pkl')
knn = joblib.load('knn_model_with_city.pkl')

# Define a function to predict the best match for keterampilan_teknis and preferred city
def predict_company(skills, user_city):
    skills_vectorized = vectorizer.transform([skills])
    city_vectorized = encoder.transform([[user_city]])  # Transform the user's city
    combined_features = hstack([skills_vectorized, city_vectorized])
    
    prediction = knn.predict(combined_features)
    return prediction[0]

# Example input
input_skills = "Penginstalan Server, Pemograman java, pemograman c, Sistem Versi Kontrol, pengembangan frontend, pengembangan backend, pengembangan website, html, css"

input_city = "bandung"
predicted_company = predict_company(input_skills, input_city)

print(f"Recommended company for PKL: {predicted_company}")
