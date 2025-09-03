from flask import Flask, render_template, request, redirect, session
import sqlite3
import pandas as pd
import pickle

# Tell Flask to use current folder for HTML & CSS
app = Flask(__name__, template_folder='.', static_folder='.')
app.secret_key = "supersecretkey"

# Load trained model
model = pickle.load(open("model.pkl", "rb"))
colleges = pd.read_csv("colleges.csv")

# ------------------ DATABASE INIT ------------------ #
def init_sqlite_db():
    conn = sqlite3.connect("database.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE,
                     password TEXT)''')
    conn.close()

init_sqlite_db()

# ------------------ ROUTES ------------------ #
@app.route("/")
def index():
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username,password) VALUES (?,?)", (username,password))
            conn.commit()
        except:
            return "⚠️ User already exists!"
        conn.close()
        return redirect("/")
    return render_template("register.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username,password))
    user = cursor.fetchone()
    conn.close()
    if user:
        session["user"] = username
        return redirect("/home")
    return "❌ Invalid Credentials!"

@app.route("/home")
def home():
    if "user" not in session:
        return redirect("/")
    return render_template("home.html")

# ------------------ RECOMMENDATION ------------------ #
@app.route("/recommend", methods=["POST"])
def recommend():
    percentile = float(request.form["percentile"])
    gender = request.form["gender"]
    category = request.form["category"]
    city = request.form["city"]
    branch = request.form["branch"]

    # Filter dataset
    subset = colleges[(colleges["Branch"]==branch) & (colleges["City"]==city)]

    if subset.empty:
        return "⚠️ No colleges found for this city/branch."

    # Prepare features
    X = subset[["Percentile","Gender","Category"]]
    X = pd.get_dummies(X)

    # Align with training model
    for col in model.feature_names_in_:
        if col not in X.columns:
            X[col] = 0
    X = X[model.feature_names_in_]

    # Predict probabilities
    probs = model.predict_proba(X)[:,1]
    subset["Admission_Probability"] = probs

    # Rank top 20
    result = subset.sort_values("Admission_Probability", ascending=False).head(20)

    return render_template("result.html", tables=[result.to_html(classes='data', header="true")])

if __name__ == "__main__":
    app.run(debug=True)
