from flask import Flask, request, jsonify
from malconv import get_prediction


# Create the Flask app
app = Flask(__name__)


# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data
    input_data = request.files["file"].read()

    # Make the prediction
    prediction = get_prediction(input_data)

    # Return the prediction as a JSON response
    return jsonify({"prediction": f"{prediction:.4f}"})


# Start the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
