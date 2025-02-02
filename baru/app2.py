import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import hstack
import joblib

df = pd.read_excel('merged_data2.xlsx')
df = df.dropna()

# Vectorize the 'kriteria_teknis' column
df = df.groupby('nama_perusahaan').agg({
    'kriteria_teknis': lambda x: ', '.join(set(', '.join(x).split(', '))),  # Concatenate unique kriteria_teknis
    # 'keterampilan': ', '.join,  # Concatenate keterampilan values
    'preferensi_lokasi': 'first',  # Concatenate preferensi_lokasi values
    # 'kriteria': 'first',  # Keep the first value of 'kriteria'
}).reset_index()
print(df)
df['kriteria_teknis'] = df['kriteria_teknis'].str.split(',').apply(lambda x: ' '.join([i.strip() for i in x]))

vectorizer = TfidfVectorizer()
X_kriteria = vectorizer.fit_transform(df['kriteria_teknis'])

# One-hot encode the company location
city_encoder = OneHotEncoder()
X_city = city_encoder.fit_transform(df[['preferensi_lokasi']])

# Combine the text features and the encoded city features
X = hstack([X_kriteria, X_city])
print("STACK: ", X)
# Use the company names as the target

label_encoder = LabelEncoder()
y = df['nama_perusahaan']
y_encoded = label_encoder.fit_transform(y)

# Train the K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y_encoded)

# Save the model, vectorizer, and encoder
joblib.dump(knn, 'knn_model_with_city.pkl')
joblib.dump(vectorizer, 'vectorizer_with_city.pkl')
joblib.dump(city_encoder, 'city_encoder_with_city.pkl')
joblib.dump(label_encoder, 'label_encoder_with_city.pkl')

# Define a function to predict the best match for keterampilan_teknis and preferred city
