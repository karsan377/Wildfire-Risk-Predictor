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

OPENWEATHER_API_KEY = 'PUT API KEY HERE'

# Load the custom wildfire model once
try:
    app.model = tf.keras.models.load_model('wildfire_transfer_model.h5')
    logger.info("Custom wildfire model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    app.model = None


def get_weather_for_city(city):
    """Fetch weather data from OpenWeather API"""
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


def preprocess_image(image_file):
    """Preprocess image to match model input requirements"""
    img = Image.open(image_file).convert('RGB')
    img = img.resize((128, 128))  # Match model input shape
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
    return img_array


def classify_image(image_file):
    """Classify image using the trained model and return the prediction score"""
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


def calculate_risk_score(weather, img_risk_score):
    """Calculate the continuous risk score using weather conditions and image risk score"""
    
    # Multipliers based on weather conditions
    temperature = weather['temp']
    wind_speed = weather['wind']
    humidity = weather['humidity']
    
    temperature_multiplier = 1 if temperature <= 21 else 1.2 if temperature <= 32 else 1.5

    # Wind speed multiplier: Stronger winds increase the risk
    # 10 mph ≈ 4.5 m/s, 30 mph ≈ 13.4 m/s
    wind_speed_multiplier = 1 if wind_speed <= 4.5 else 1.2 if wind_speed <= 13.4 else 1.5

    # Humidity multiplier: Low humidity increases the risk
    humidity_multiplier = 1 if humidity >= 40 else 1.2 if humidity >= 30 else 1.5

    # Calculate composite risk score
    risk_score = img_risk_score * temperature_multiplier * wind_speed_multiplier * humidity_multiplier
    return risk_score


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

        img_risk_score = img_result  # This is a float between 0 and 1 (probability of wildfire risk)

        # Calculate continuous risk score based on weather multipliers and image score
        final_risk_score = calculate_risk_score(weather, img_risk_score)

        # Define risk levels based on the final score
        if final_risk_score >= 2:
            risk = "Extreme ⚠️"
        elif final_risk_score >= 1.2:
            risk = "High 🔥"
        elif final_risk_score >= 0.7:
            risk = "Moderate ⚠️"
        else:
            risk = "Low "

        logger.info(f"Risk assessment complete for {city}: {risk}")
        return jsonify({
            "fire_risk": risk,
            "details": {
                "city": city,
                "temperature": weather['temp'],
                "humidity": weather['humidity'],
                "wind_speed": weather['wind'],
                "image_risk_score": img_risk_score,
                "final_risk_score": final_risk_score
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

        # Actually validate the city by trying to get weather data
        weather = get_weather_for_city(city)
        if "error" in weather:
            return jsonify({"error": weather["error"]}), 400

        return jsonify({'message': 'City verified successfully'})

    except Exception as e:
        logger.error(f"Error verifying city: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
