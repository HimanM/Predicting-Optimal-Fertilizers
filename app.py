from flask import Flask, request, jsonify
from Predictor import FertilizerRecommender  # Adjust import path

app = Flask(__name__)

# Initialize model once at startup
recommender = FertilizerRecommender()

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            return jsonify({"error": "No JSON data provided or data is not a valid object."}), 400
        
        # Validate required fields exist
        required_fields = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        
        # Get top 3 fertilizer prediction string
        top3_list = recommender.predict_top3(data) # This returns names only

        # Also get probabilities
        top3_with_probs = recommender.predict(data, top_k=3) # This returns list of (fertilizer, prob)

        # Ensure it's a list, if it's a string, split it
        if isinstance(top3_list, str):
            top3_list = top3_list.split()

        # Join the list into a comma-separated string
        top3_comma_separated_str = ",".join(top3_list)

        # Format predictions with probabilities
        predictions_detailed = []
        if isinstance(top3_with_probs, list) and len(top3_with_probs) > 0 and isinstance(top3_with_probs[0], tuple):
            # Single input
            predictions_detailed = [
                {"fertilizer": fert, "probability": float(prob)} for fert, prob in top3_with_probs
            ]
        elif isinstance(top3_with_probs, list) and len(top3_with_probs) > 0 and isinstance(top3_with_probs[0], list):
            # Batch input (not expected here, but handle for robustness)
            predictions_detailed = [
                [
                    {"fertilizer": fert, "probability": float(prob)} for fert, prob in row
                ] for row in top3_with_probs
            ]

        # Return as JSON
        return jsonify({
            "predictions": top3_comma_separated_str,
            "predictions_detailed": predictions_detailed
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
