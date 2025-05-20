from flask import Flask, request, render_template, jsonify
from app.inference import predict_image
from app.utils import setup_logging
import os
import traceback
import sys

app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
setup_logging()

@app.route('/')
def index():
    app.logger.info("📥 GET / accessed")
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        front_file = request.files.get("front")
        back_file = request.files.get("back")

        results = {}
        if front_file:
            app.logger.info("📸 Received front image for prediction.")
            label, gradcam_url = predict_image(front_file, side="front")
            results['front_prediction'] = label
            results['front_gradcam'] = gradcam_url
            app.logger.info(f"✅ Front prediction: {label}")

        if back_file:
            app.logger.info("📸 Received back image for prediction.")
            label, gradcam_url = predict_image(back_file, side="back")
            results['back_prediction'] = label
            results['back_gradcam'] = gradcam_url
            app.logger.info(f"✅ Back prediction: {label}")

        if not results:
            app.logger.warning("⚠️ No files uploaded in request.")
            return jsonify({"error": "No files uploaded"}), 400

        return jsonify(results)

    except Exception as e:
        app.logger.error("🔥 Exception in /predict route:")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
