import pickle
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(".", "index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "LogisticRegression + TF-IDF"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Field 'text' wajib diisi"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text tidak boleh kosong"}), 400

        proba = model.predict_proba([text])[0]
        label = model.predict([text])[0]

        result = "real" if label == 1 else "fake"
        confidence = round(float(max(proba)) * 100, 2)

        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "status": "sukses"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
