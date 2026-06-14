# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:00:51 2024

@author: uktup
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open(r"C:\Upendra\ML-MODEL-DEPLOYMENT-USING-FLASK-main (3) (1)\ML-MODEL-DEPLOYMENT-USING-FLASK-main\model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=False)
    
import os
os.getcwd()