import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load CSV
df = pd.read_csv("students_colleges.csv")  # your CSV filename

# Features and target
# Features: Percentile, Gender, Category
X = df[["Percentile","Gender","Category"]]
X = pd.get_dummies(X)  # Convert categorical columns to numbers

# Target: Admitted or not (1/0)
# Assuming you have column "Admitted" or create one based on logic
# Example: top colleges based on percentile for each branch/city
if "Admitted" not in df.columns:
    # Example logic: if percentile > 90, mark as admitted
    X_target = df.copy()
    X_target["Admitted"] = df["Percentile"].apply(lambda x: 1 if x >= 90 else 0)
    y = X_target["Admitted"]
else:
    y = df["Admitted"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")
