from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__, template_folder='.', static_folder='.')

# Load trained model and dataset
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("students_colleges.csv")

# ------------------ ROUTES ------------------ #
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get user input
    percentile = float(request.form["percentile"])
    gender = request.form["gender"]
    category = request.form["category"]
    city = request.form["city"]
    branch = request.form["branch"]

    # Filter dataset by city & branch
    subset = df[(df["Branch"]==branch) & (df["City"]==city)]
    if subset.empty:
        return "⚠️ No colleges found for this city/branch."

    # Prepare features
    X_user = subset[["Percentile", "Gender", "Category"]]
    X_user = pd.get_dummies(X_user)

    # Add missing columns (align with model)
    for col in model.feature_names_in_:
        if col not in X_user.columns:
            X_user[col] = 0
    X_user = X_user[model.feature_names_in_]

    # Predict admission probability
    subset["Admission_Probability"] = model.predict(X_user)

    # Rank top 20
    result = subset.sort_values("Admission_Probability", ascending=False).head(20)
    predicted_count = result.shape[0]

    return render_template("result.html", tables=[result.to_html(classes='data', header="true")], count=predicted_count)

if __name__ == "__main__":
    app.run(debug=True)
