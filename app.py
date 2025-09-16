from flask import Flask, render_template, request, redirect, session
import sqlite3
import pandas as pd
import pickle

app = Flask(__name__, template_folder='.', static_folder='.')
app.secret_key = "supersecretkey"

# Load model and CSV
model = pickle.load(open("model.pkl","rb"))
df = pd.read_csv("students_colleges.csv")

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("database.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE,
                     password TEXT)''')
    conn.close()

init_db()

# ---------------- ROUTES ----------------
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

# ---------------- RECOMMENDATION ----------------
@app.route("/recommend", methods=["POST"])
def recommend():
    percentile = float(request.form["percentile"])
    gender = request.form["gender"]
    category = request.form["category"]
    city = request.form["city"]
    branch = request.form["branch"]

    # Filter dataset
    subset = df[(df["Branch"]==branch) & (df["City"]==city)]

    if subset.empty:
        return "⚠️ No colleges found for this city/branch."

    # Prepare features for prediction
    X_user = subset[["Percentile","Gender","Category"]]
    X_user = pd.get_dummies(X_user)

    # Align columns with trained model
    for col in model.feature_names_in_:
        if col not in X_user.columns:
            X_user[col] = 0
    X_user = X_user[model.feature_names_in_]

    # Predict admission probability
    subset["Admission_Probability"] = model.predict_proba(X_user)[:,1]

    # Top 20 recommendations
    result = subset.sort_values("Admission_Probability", ascending=False).head(20)

    return render_template("result.html", tables=[result.to_html(classes='data', header="true")])

if __name__ == "__main__":
    app.run(debug=True)
