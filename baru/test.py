import joblib

knn = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define a function to predict the best match for keterampilan_teknis and return the company name
def predict_company(skills):
    skills_vectorized = vectorizer.transform([skills])
    prediction = knn.predict(skills_vectorized)
    print(prediction)
    return prediction[0]

# Example input
input_skills = "pengembangan website, basis data, pemograman mobile, desain ui/ux, instalasi jaringan, konfigurasi jaringan"
predicted_company = predict_company(input_skills)
print(f"Recommended company for PKL: {predicted_company}")
