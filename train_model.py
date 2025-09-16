import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("students_colleges.csv")

# Features
X = df[["Percentile", "Gender", "Category"]]
X = pd.get_dummies(X)  # Convert categorical variables to numeric

# Target
# If you don't have historical admitted data, use percentile as proxy
# This allows model to predict probability for all students
y = df["Percentile"] / 100.0  # Normalize percentile to 0-1 for probability

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("✅ Model trained and saved as model.pkl")
