from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import tensorflow as tf
import numpy as np
import logging
import io

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY = 'c4d1b1586d9f035fc90c53721dcfd8cd'

# Load the custom wildfire model once
try:
    app.model = tf.keras.models.load_model('wildfire_cnn_model.h5')
    logger.info("Custom wildfire model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    app.model = None


def get_weather_for_city(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        res = response.json()
        
        if 'main' not in res:
            logger.error(f"Unexpected weather API response: {res}")
            return {"error": "Invalid weather data received"}
            
        return {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "wind": res["wind"]["speed"]
        }
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return {"error": f"Weather service unavailable: {str(e)}"}


def preprocess_image(image_file, target_size=(64, 64)):
    """Resize, normalize, and expand dimensions of image for CNN model."""
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)  # Shape: (1, 64, 64, 3)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


def classify_image(image_file):
    if app.model is None:
        return {"error": "Model not available"}
    try:
        processed_image = preprocess_image(image_file)
        prediction = app.model.predict(processed_image)[0][0]
        logger.debug(f"Model raw prediction: {prediction}")
        return float(prediction)  # Probability of wildfire risk
    except Exception as e:
        logger.error(f"Image classification error: {str(e)}")
        return {"error": f"Image classification failed: {str(e)}"}


@app.route('/assess-risk', methods=['POST'])
def assess_risk():
    try:
        logger.debug("Received risk assessment request")

        city = request.form.get('city')
        if not city:
            return jsonify({"error": "City name is required"}), 400

        image = request.files.get('image')
        if not image:
            return jsonify({"error": "Image is required"}), 400

        weather = get_weather_for_city(city)
        if "error" in weather:
            return jsonify({"error": weather["error"]}), 400

        img_result = classify_image(image)
        if isinstance(img_result, dict) and "error" in img_result:
            return jsonify({"error": img_result["error"]}), 400

        img_risk_score = img_result  # a float between 0 and 1
        image_risk = 1 if img_risk_score > 0.5 else 0

        score = image_risk
        score += 1 if weather['humidity'] < 30 else 0
        score += 1 if weather['temp'] > 30 else 0
        score += 1 if weather['wind'] > 10 else 0

        if score >= 4:
            risk = "Extreme ⚠️"
        elif score == 3:
            risk = "High 🔥"
        elif score == 2:
            risk = "Moderate ⚠️"
        else:
            risk = "Low ✅"

        logger.info(f"Risk assessment complete for {city}: {risk}")
        return jsonify({
            "fire_risk": risk,
            "details": {
                "city": city,
                "temperature": weather['temp'],
                "humidity": weather['humidity'],
                "wind_speed": weather['wind'],
                "image_risk_score": img_risk_score
            }
        })

    except Exception as e:
        logger.error(f"Unexpected error in risk assessment: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
