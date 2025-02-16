import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_model():
    # Load dataset and define features and target
    data = pd.read_csv('life_insurance_prediction.csv')
    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type']
    target = 'Prediction_Target'

    # Prepare data for training
    X = data[features]
    y = data[target]

    # Standardize the categorical columns to have consistent case
    X.loc[:, 'Gender'] = X['Gender'].str.capitalize()
    X.loc[:, 'Health_Status'] = X['Health_Status'].str.capitalize()
    X.loc[:, 'Smoking_Habit'] = X['Smoking_Habit'].str.capitalize()
    X.loc[:, 'Family_History'] = X['Family_History'].str.capitalize()
    X.loc[:, 'Policy_Type'] = X['Policy_Type'].str.capitalize()

    # Label encode categorical columns
    label_encoders = {}
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type']:
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Train the premium model (without premium_amount as input feature)
    premium_model = RandomForestClassifier()
    premium_model.fit(X_train, data.loc[X_train.index, 'Premium_Amount'])

    return model, premium_model, label_encoders, accuracy

def get_user_input():
    # Collect user inputs (excluding premium_amount)
    age = int(input("Enter age: "))
    gender = input("Enter gender (Male/Female): ").capitalize()  # Ensure consistent capitalization
    income = float(input("Enter income: "))
    health_status = input("Enter health status (Excellent/Good/Average/Poor): ").capitalize()
    smoking = input("Do you smoke? (Yes/No): ").capitalize()
    family_history = input("Do you have a family history of illness? (Yes/No): ").capitalize()

    return age, gender, income, health_status, smoking, family_history

def predict_insurance():
    # Get user inputs
    age, gender, income, health_status, smoking, family_history = get_user_input()

    # Train the model every time (since we're not using pickle)
    model, premium_model, label_encoders, accuracy = train_model()

    # Prepare input data for prediction (without premium_amount)
    input_data = pd.DataFrame([[age, gender, income, health_status, smoking, family_history, 'None']],
                               columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Family_History', 'Policy_Type'])

    # Encode categorical inputs
    for col in label_encoders:
        # Only transform columns that have valid label encoding
        if col != 'Policy_Type':  # Don't encode 'Policy_Type' yet
            input_data.loc[:, col] = label_encoders[col].transform(input_data[col])

    # Ensure 'Policy_Type' column is handled correctly, as it is not used during encoding
    input_data['Policy_Type'] = 0  # You can assign any placeholder integer (e.g., 0)

    # Make prediction for eligibility
    prediction = model.predict(input_data)

    # Determine eligible policies based on conditions
    if income > 100000 and health_status == 'Excellent':
        eligible_policies = ['Whole', 'Universal', 'Term']
    elif income > 50000 and health_status in ['Good', 'Average']:
        eligible_policies = ['Universal', 'Term']
    else:
        eligible_policies = ['Term']

    # Estimate premiums for eligible policies
    premium_estimates = {}
    for policy in eligible_policies:
        input_data['Policy_Type'] = label_encoders['Policy_Type'].transform([policy])[0]
        premium_estimates[policy] = premium_model.predict(input_data)[0]

    # Return the result
    result = 'Eligible' if prediction[0] == 1 else 'Not Eligible'
    return result, eligible_policies, premium_estimates, accuracy

# Example usage
if __name__ == "__main__":
    result, eligible_policies, premium_estimates, accuracy = predict_insurance()
    print(f"Eligibility: {result}")
    print(f"Eligible Policies: {eligible_policies}")
    print(f"Premium Estimates: {premium_estimates}")
    print(f"Model Accuracy: {accuracy:.2%}")
