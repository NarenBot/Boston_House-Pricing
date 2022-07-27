import pickle
import re
import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, jsonify, url_for, request

model = pickle.load(open("GradModel.pkl", "rb"))
scaler = pickle.load(open("Scaling.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def postman():
    data = request.json['data']
    print(data)
    new_data = np.array(list(data.values())).reshape(1,-1)
    print(new_data)
    scaled_data = scaler.transform(new_data)
    output = model.predict(scaled_data)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [ float(x) for x in request.form.values()]
    new_data = np.array(list(data)).reshape(1,-1)
    scaled_data = scaler.transform(new_data)
    output = model.predict(scaled_data)
    return render_template('home.html', prediction_text = f"House Price Prediction is: {output[0]}")



if __name__ == '__main__':
    app.run(debug=True)
