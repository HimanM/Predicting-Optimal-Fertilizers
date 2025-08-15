from flask import Flask, request, jsonify
from Predictor import FertilizerRecommender

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
    import os

    # Read runtime configuration from environment variables so this app
    # can run safely under a production WSGI server (gunicorn) or standalone.
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 8000))

    # Default to production environment unless explicitly enabled for debug
    if not debug:
        app.config.update(DEBUG=False, ENV="production")

    app.run(host=host, port=port, debug=debug)
