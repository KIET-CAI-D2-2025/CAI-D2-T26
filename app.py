from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

def train_model():
    data = pd.read_csv('life_insurance_prediction.csv')
    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type']
    target = 'Prediction_Target'
    
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
        data[col] = data[col].str.capitalize()
    
    label_encoders = {}
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    premium_model = RandomForestClassifier(random_state=42)
    premium_model.fit(X_train, data.loc[X_train.index, 'Premium_Amount'])
    
    return model, premium_model, label_encoders, accuracy

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = int(data['age'])
    gender = data['gender'].capitalize()
    income = float(data['income'])
    health_status = data['health'].capitalize()
    smoking = data['smoke'].capitalize()
    
    model, premium_model, label_encoders, accuracy = train_model()
    
    input_data = pd.DataFrame([[age, gender, income, health_status, smoking, 'None']],
                               columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type'])
    
    for col in label_encoders:
        if col != 'Policy_Type':
            input_data[col] = label_encoders[col].transform(input_data[col])
    
    input_data['Policy_Type'] = 0
    
    prediction = model.predict(input_data)
    response = {}
    
    if income > 100000 and health_status == 'Excellent':
        eligible_policies = ['Whole', 'Universal', 'Term']
    elif income > 50000 and health_status in ['Good', 'Average']:
        eligible_policies = ['Universal', 'Term']
    elif income > 5000:
        eligible_policies = ['Term']
    else:
        response = {
            'eligible': False,
            'message': 'Sorry, you are not eligible for life insurance at this time.',
            'model_accuracy': accuracy
        }
        return jsonify(response)
    
    premium_estimates = {}
    for policy in eligible_policies:
        input_data['Policy_Type'] = label_encoders['Policy_Type'].transform([policy])[0]
        premium_estimates[policy] = float(premium_model.predict(input_data)[0])
    
    response = {
        'eligible': True,
        'message': 'You are eligible for life insurance!',
        'policies': eligible_policies,
        'premiums': premium_estimates,
        'model_accuracy': accuracy
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
