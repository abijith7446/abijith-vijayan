import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset (relative to the root 'netflix' directory)
df = pd.read_csv("data set/netflix_titles.csv")

# 2. Keep required columns and drop missing values
df = df[['type', 'country', 'listed_in', 'duration']].dropna()

# 3. Convert duration to numbers
df['duration'] = df['duration'].str.extract('(\d+)').astype(int)

# 4. Encode categorical columns
country_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df['country'] = country_encoder.fit_transform(df['country'])
df['listed_in'] = genre_encoder.fit_transform(df['listed_in'])

# 5. Split features and target
X = df[['country', 'listed_in', 'duration']]
y = df['type']

# 6. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 7. Save models using joblib into the 'notebook' folder
joblib.dump(model, "notebook/classification_model.pkl")
joblib.dump(country_encoder, "notebook/country_encoder.pkl")
joblib.dump(genre_encoder, "notebook/genre_encoder.pkl")

print("All models trained and saved successfully inside the 'notebook' folder!")