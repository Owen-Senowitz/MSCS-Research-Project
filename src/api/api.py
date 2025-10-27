from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import os
from tensorflow import keras
import tensorflow as tf
import base64


app = Flask(__name__)
CORS(app)

# Model paths
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
MODEL_FILES = {
	"best_b3": "best_b3.keras",
	"best_cnn": "best_cnn.keras",
	"best_efficientnet": "best_efficientnet.keras",
	"best_transunet": "best_transunet.keras",
}

# Cache loaded models
loaded_models = {}

def load_model(model_name):
	if model_name not in MODEL_FILES:
		raise ValueError(f"Model '{model_name}' not found.")
	if model_name not in loaded_models:
		model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
		loaded_models[model_name] = keras.models.load_model(model_path)
	return loaded_models[model_name]

def preprocess_image(image_bytes, input_shape):
	print(f"[DEBUG] Uploaded image bytes length: {len(image_bytes)}")
	try:
		image = Image.open(io.BytesIO(image_bytes))
	except Exception as e:
		print(f"[DEBUG] Pillow failed to open image: {e}")
		return None
	# input_shape: (None, height, width, channels)
	height, width, channels = input_shape[1:]
	if channels == 1:
		image = image.convert("L")  # grayscale
	else:
		image = image.convert("RGB")
	image = image.resize((width, height))
	arr = np.array(image) / 255.0
	if channels == 1:
		arr = np.expand_dims(arr, axis=-1)
	arr = np.expand_dims(arr, axis=0)
	return arr

@app.route("/predict", methods=["POST"])
def predict():
	try:
		if "file" not in request.files or "model_name" not in request.form:
			return jsonify({"success": False, "error": "Missing file or model_name"}), 400
		file = request.files["file"]
		model_name = request.form["model_name"]
		image_bytes = file.read()
		model = load_model(model_name)
		input_shape = model.input_shape  # (None, height, width, channels)
		img = preprocess_image(image_bytes, input_shape)
		if img is None:
			return jsonify({"success": False, "error": "Invalid or empty image uploaded."}), 400
		preds = model.predict(img)
		result = preds.tolist()

		# Output prediction and label (benign/malignant) only
		label = None
		if model_name == "best_transunet":
			pred_value = float(result[0][0]) if isinstance(result[0], list) else float(result[0])
			label = "Malignant" if pred_value > 0.5 else "Benign"
		return jsonify({
			"success": True,
			"prediction": result,
			"label": label
		})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500

@app.route("/", methods=["GET"])
def root():
	return jsonify({"message": "ML Model API is running."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
