import joblib 
from flask import Flask, request, render_template, jsonify, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

# load the model
model = joblib.load(open('model/model.joblib', 'rb'))
pipeline = joblib.load(open('model/pipeline.joblib', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    final_input = pipeline.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template("index.html", prediction=f"The House Price Prediction is {output}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    dataArr = np.array(list(data.values())).reshape(1, -1)
    print(dataArr)
    new_data = pipeline.transform(dataArr)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])
 
if __name__ == "__main__":
    app.run(debug=True)