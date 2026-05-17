from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("iris_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["sepal_length"]),
        float(request.form["sepal_width"]),
        float(request.form["petal_length"]),
        float(request.form["petal_width"])
    ]

    prediction = model.predict([features])

    species = ["Setosa", "Versicolor", "Virginica"]

    result = species[prediction[0]]

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)