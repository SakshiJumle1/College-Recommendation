import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("data/colleges.csv")

# Features & Target
X = df[["Percentile","Gender","Category"]]
X = pd.get_dummies(X)
y = df["Admitted"]   # 1 = admitted, 0 = not admitted

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl","wb"))

print("✅ Model trained and saved as model.pkl")
