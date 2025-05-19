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
            label, gradcam_url = predict_image(front_file, side="front")
            results['front_prediction'] = label
            results['front_gradcam'] = gradcam_url

        if back_file:
            label, gradcam_url = predict_image(back_file, side="back")
            results['back_prediction'] = label
            results['back_gradcam'] = gradcam_url

        return jsonify(results)

    except Exception as e:
        import traceback
        app.logger.error("🔥 Exception in /predict route:")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)