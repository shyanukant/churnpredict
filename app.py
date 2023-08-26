import joblib 
from flask import Flask, request, render_template, jsonify, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
app.debug = True
# load the model
model = joblib.load(open('model/model.joblib', 'rb'))
pipeline = joblib.load(open('model/pipeline.joblib', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    result = ''
    age = float(request.form['Age'])
    gender = request.form['Gender']
    location = request.form['Location']
    subscription_Length_Months = float(request.form['Subscription_Length_Months'])
    monthly_Bill = float(request.form['Monthly_Bill'])
    total_Usage_GB = float(request.form['Total_Usage_GB'])

    data = [[age, gender, location, subscription_Length_Months, monthly_Bill, total_Usage_GB]]
    print(data)
    user_input = pd.DataFrame(data , columns=['Age', 'Gender', 'Location', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB'])
    # print(user_input)
    final_input = pipeline.transform(user_input)
    # print(final_input)
    output = model.predict(final_input)[0]
    if output == 0:
        result = 'not churn'
    else:
        result = 'churn'
    return render_template("index.html", prediction=f"The Customer will {result}")
 
if __name__ == "__main__":
    app.run()