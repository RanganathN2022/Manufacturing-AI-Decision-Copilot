from flask import Flask, request, jsonify

from app.predict import predict_failure


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Manufacturing AI Decision Copilot",
        "status": "running",
        "service_type": "Machine Failure Prediction API"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        machine_data = request.get_json()

        if not machine_data:
            return jsonify({
                "error": "No machine data provided"
            }), 400

        result = predict_failure(
            machine_data
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
    