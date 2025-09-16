from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from PIL import Image
import tensorflow as tf
import numpy as np
import logging
import io
import os

app = Flask(__name__, template_folder='templates')  # Make sure your HTML is in templates/
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY = 'c4d1b1586d9f035fc90c53721dcfd8cd'

# Load the custom wildfire model once
try:
    app.model = tf.keras.models.load_model('wildfire_transfer_model.h5')
    logger.info("Custom wildfire model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    app.model = None


# ------------------ Utility Functions (unchanged) ------------------
def get_weather(lat=None, lon=None, city=None):
    """Fetch weather data from OpenWeather API by coordinates or city name"""
    try:
        if lat is not None and lon is not None:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        elif city:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        else:
            logger.error("No location provided for weather lookup")
            return {"error": "No location provided"}

        response = requests.get(url)
        response.raise_for_status()
        res = response.json()

        if 'main' not in res:
            logger.error(f"Unexpected weather API response: {res}")
            return {"error": "Invalid weather data received"}

        return {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "wind": res["wind"]["speed"],
            "city": res.get("name", city)
        }
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        return {"error": f"Weather service unavailable: {str(e)}"}


def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def classify_image(image_file):
    if app.model is None:
        return {"error": "Model not available"}
    try:
        processed_image = preprocess_image(image_file)
        prediction = app.model.predict(processed_image)[0][0]
        logger.debug(f"Model raw prediction: {prediction}")
        return float(prediction)
    except Exception as e:
        logger.error(f"Image classification error: {str(e)}")
        return {"error": f"Image classification failed: {str(e)}"}


def calculate_risk_score(weather, img_risk_score):
    temperature = weather['temp']
    wind_speed = weather['wind']
    humidity = weather['humidity']

    temperature_multiplier = 1 if temperature <= 21 else 1.2 if temperature <= 32 else 1.3
    wind_speed_multiplier = 1 if wind_speed <= 3.5 else 1.2 if wind_speed <= 8 else 1.5
    humidity_multiplier = 1 if humidity >= 40 else 1.1 if humidity >= 30 else 1.2

    return img_risk_score * temperature_multiplier * wind_speed_multiplier * humidity_multiplier

# ------------------ Frontend Route ------------------
@app.route('/')
def index():
    """Serve the frontend HTML"""
    return render_template('index.html')


# ------------------ API Endpoints (unchanged) ------------------
@app.route('/assess-risk', methods=['POST'])
def assess_risk():
    try:
        logger.debug("Received risk assessment request")

        city = request.form.get('city')
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')

        if lat and lon:
            try:
                lat = float(lat)
                lon = float(lon)
            except ValueError:
                return jsonify({"error": "Invalid latitude or longitude"}), 400
            weather = get_weather(lat=lat, lon=lon)
        elif city:
            weather = get_weather(city=city)
        else:
            return jsonify({"error": "City or coordinates are required"}), 400

        if "error" in weather:
            return jsonify({"error": weather["error"]}), 400

        image = request.files.get('image')
        if not image:
            return jsonify({"error": "Image is required"}), 400

        img_result = classify_image(image)
        if isinstance(img_result, dict) and "error" in img_result:
            return jsonify({"error": img_result["error"]}), 400

        img_risk_score = img_result
        final_risk_score = calculate_risk_score(weather, img_risk_score)

        if final_risk_score >= 2:
            risk = "Extreme"
        elif final_risk_score >= 1.2:
            risk = "High"
        elif final_risk_score >= 0.7:
            risk = "Moderate"
        else:
            risk = "Low"

        logger.info(f"Risk assessment complete for {weather.get('city', city)}: {risk}")
        return jsonify({
            "fire_risk": risk,
            "details": {
                "city": weather.get("city", city),
                "temperature": weather['temp'],
                "humidity": weather['humidity'],
                "wind_speed": weather['wind'],
                "image_risk_score": round(img_risk_score, 3),
                "final_risk_score": round(final_risk_score, 3)
            }
        })

    except Exception as e:
        logger.error(f"Unexpected error in risk assessment: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/verify-city', methods=['POST'])
def verify_city():
    try:
        data = request.get_json()
        city = data.get('city', '').strip()

        if not city:
            return jsonify({'error': 'City name is required'}), 400

        weather = get_weather(city=city)
        if "error" in weather:
            return jsonify({"error": weather["error"]}), 400

        return jsonify({'message': 'City verified successfully'})

    except Exception as e:
        logger.error(f"Error verifying city: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


# ------------------ Run App ------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's port if available, else 5000
    app.run(host='0.0.0.0', port=port, debug=True)