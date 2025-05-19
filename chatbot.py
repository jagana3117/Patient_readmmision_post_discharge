from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Setup
app = Flask(__name__)

# Load environment variable
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in environment!")
    genai_model = None
else:
    print("✅ GOOGLE_API_KEY loaded successfully.")
    genai.configure(api_key=api_key)
    genai_model = genai.GenerativeModel("gemini-pro")

# Route for chatbot messages
@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("message", "")

    if not user_input:
        return jsonify({"reply": "Sorry, I didn't catch that. Please try again."})

    if not genai_model:
        return jsonify({"reply": "API key not configured. Please check the server setup."})

    try:
        prompt = f"""
        You are Health Pulse Support Bot, a helpful assistant in a hospital setting. 
        Answer questions about post-discharge care, medication, symptoms, hospital services, and health-related concerns clearly and compassionately.

        User: {user_input}
        """
        response = genai_model.generate_content(prompt)
        return jsonify({"reply": response.text.strip()})

    except Exception as e:
        print("Chatbot error:", e)
        return jsonify({"reply": "Sorry, there was a problem generating the response."})

# Run server
if __name__ == "__main__":
    app.run(debug=True)
