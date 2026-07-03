from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ---------------- Load Model ----------------
model = joblib.load(r"C:\Users\mayur\OneDrive\Desktop\data science training\project\medical\model\rfc_load.lb")


# ---------------- Disease Mapping ----------------
disease_map = {
    1: "Healthy",
    2: "Pre-Diabetes",
    3: "Hypertension",
}


# ---------------- INFO ----------------
@app.route('/')
def info():
    return render_template("info.html")


# ---------------- HOME ----------------
@app.route('/home')
def home():
    return render_template("home.html")


# ---------------- ABOUT ----------------
@app.route('/about')
def about():
    return render_template("about.html")


# ---------------- CONTACT ----------------
@app.route('/contact', methods=['GET', 'POST'])
def contact():

    if request.method == "POST":

        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        subject = request.form['subject']
        address = request.form['address']
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        pincode = request.form['pincode']
        message = request.form['message']

        print(name)
        print(email)

        return render_template(
            "contact.html",
            success="Message Sent Successfully!"
        )

    return render_template("contact.html")


# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['POST'])
def predict():

    # -------- Read Form --------

    gender = request.form["Gender"]
    smoking = request.form["Smoking"]
    alcohol = request.form["Alcohol"]
    physical = request.form["PhysicalActivity"]
    family = request.form["FamilyHistory"]

    age = float(request.form["Age"])
    bmi = float(request.form["BMI"])
    bp = float(request.form["BloodPressure"])
    glucose = float(request.form["GlucoseLevel"])
    cholesterol = float(request.form["Cholesterol"])
    heart = float(request.form["HeartRate"])


    # -------- Encoding --------

    gender = 1 if gender == "Male" else 0

    smoking = 1 if smoking == "Yes" else 0

    alcohol = 1 if alcohol == "Yes" else 0

    family = 1 if family == "Yes" else 0

    if physical == "Low":
        physical = 0
    elif physical == "Moderate":
        physical = 1
    else:
        physical = 2


    # -------- Feature Order --------

    features = np.array([[

        gender,
        smoking,
        alcohol,
        physical,
        family,
        age,
        bmi,
        bp,
        glucose,
        cholesterol,
        heart

    ]])


    # -------- Prediction --------

    pred = model.predict(features)[0]

    print("Predicted Class :", pred)


    # -------- Disease Name --------

    disease_name = disease_map.get(int(pred), "Unknown Disease")


    print("Disease :", disease_name)


    return render_template(
        "predict.html",
        prediction_text=disease_name
    )


if __name__ == "__main__":
    app.run(debug=True)











