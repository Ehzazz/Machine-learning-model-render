import joblib  # Make sure to have joblib installed or use pickle if that’s what you prefer
from flask import Flask, request, jsonify

# Load your trained model (change 'your_model.pkl' to your actual model filename)
model = joblib.load("model.pkl")

# Create the Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data as JSON
        data = request.json

        # Extract features from the input JSON
        feature1 = float(data.get("feature1"))
        feature2 = float(data.get("feature2"))
        feature3 = float(data.get("feature3"))
        feature4 = float(data.get("feature4"))
        feature5 = float(data.get("feature5"))

        # Create a feature array from the input data
        features = [[feature1, feature2, feature3, feature4, feature5]]
        
        # Make prediction using the model
        prediction = model.predict(features)

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction[0]})  # Assuming prediction is a single value
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Optional: Define a root endpoint to check if the API is running
@app.route("/")
def home():
    return {"message": "Welcome to the Prediction API!"}


if __name__ == "__main__":
    app.run(debug=True)
