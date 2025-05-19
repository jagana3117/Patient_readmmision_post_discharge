from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print("API Key Loaded:", bool(api_key))  # Debug print

# Configure Gemini
if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
else:
    gemini_model = None
    print("❌ Google API key not found!")

# Load ML model components
scalers = joblib.load("scalers.pkl")
label_encoders = joblib.load("label_encoders.pkl")
model = joblib.load("xgboost_patient_readmission.pkl")

categorical_columns = ['age', 'medical_specialty', 'change', 'diabetes_med']
numerical_columns = ['n_lab_procedures', 'n_inpatient', 'n_emergency']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/patient-readmission', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect data
            data = {
                'age': request.form['age'],
                'n_lab_procedures': int(request.form['n_lab_procedures']),
                'n_inpatient': int(request.form['n_inpatient']),
                'n_emergency': int(request.form['n_emergency']),
                'medical_specialty': request.form['medical_specialty'],
                'change': request.form['change'],
                'diabetes_med': request.form['diabetes_med']
            }
            df = pd.DataFrame([data])
            
            # Encode categoricals
            for col in categorical_columns:
                if col in label_encoders:
                    unseen = df[col][~df[col].isin(label_encoders[col].classes_)]
                    if not unseen.empty:
                        label_encoders[col].classes_ = np.append(label_encoders[col].classes_, unseen.unique())
                    df[col] = df[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

            # Scale numericals
            for col in numerical_columns:
                if col in scalers:
                    df[col] = scalers[col].transform(df[[col]])

            # Predict
            predictions = model.predict(df)
            if 'readmitted' in label_encoders:
                result = label_encoders['readmitted'].inverse_transform(predictions)[0]
            else:
                result = predictions[0]

            return render_template('index.html', prediction=result)

        except Exception as e:
            print("Error:", e)
            return render_template('index.html', prediction="Error in processing request")

    return render_template('index.html', prediction=None)

@app.route('/post-discharge')
def post_discharge():
    return render_template('post-discharge.html')

@app.route('/dashboard')
def dashboard():
    return render_template('graphs.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    user_input = request.json.get("message", "").strip()

    if not user_input:
        return jsonify({"reply": "Sorry, I didn't catch that. Please try again."})

    if not gemini_model:
        return jsonify({"reply": "API key not configured properly. Please check server setup."})

    try:
        prompt = f"""
        You are Health Pulse Support Bot, a helpful assistant in a hospital setting. 
        Answer questions related to:
        - Post-discharge care
        - Patient readmission
        - Medication and symptoms
        - Hospital services

        Respond with clarity, professionalism, and empathy.

        User: {user_input}
        """
        response = gemini_model.generate_content(prompt)
        return jsonify({"reply": response.text.strip()})
    
    except Exception as e:
        print("Chatbot error:", e)
        return jsonify({"reply": "Sorry, there was a problem generating the response."})

@app.route('/chatbot.html')
def chatbot_page():
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
