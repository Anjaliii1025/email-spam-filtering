from flask import Flask, render_template, request
from functions import (load_model, predict_spam)

app = Flask(__name__)

@app.route("/email")
def email():
    return render_template("index.html", spam_prediction=None, error=None)


@app.route("/emailpredict", methods=["POST"])
def predict():
    if request.method == "POST":
        email_content = request.form["email_content"]
        try:
            model, vectorizer = load_model()
            prediction = predict_spam(email_content, model, vectorizer)
            return render_template("index.html", spam_prediction=prediction, error=None)
        except Exception as e:
            return render_template("index.html", spam_prediction=None, error=str(e))
