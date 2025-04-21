from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from PIL import Image
import tensorflow as tf
import numpy as np
import logging
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY = 'c4d1b1586d9f035fc90c53721dcfd8cd'

def get_weather_for_city(city):
    """Get weather data from OpenWeather API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()  # Raises exception for 4XX/5XX status codes
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

def classify_image(image_file):
    """Classify image and return risk score"""
    try:
        # Read image directly from file stream
        img = Image.open(io.BytesIO(image_file.read()))
        img = img.convert('RGB').resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Load model (cache it after first load)
        if not hasattr(app, 'model'):
            app.model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        predictions = app.model.predict(img_array)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

        risky_labels = ["hay", "thatch", "meadow", "forest", "volcano", "cliff", "valley"]
        risk_score = sum(1 for label in decoded if any(risky in label[1] for risky in risky_labels))
        
        logger.debug(f"Image classification results: {decoded}")
        return risk_score
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return {"error": f"Image processing failed: {str(e)}"}

@app.route('/assess-risk', methods=['POST'])
def assess_risk():
    try:
        logger.debug("Received risk assessment request")
        
        # Get form data
        city = request.form.get('city')
        if not city:
            return jsonify({"error": "City name is required"}), 400
            
        image = request.files.get('image')
        if not image:
            return jsonify({"error": "Image is required"}), 400

        # Get weather data
        weather = get_weather_for_city(city)
        if "error" in weather:
            return jsonify({"error": weather["error"]}), 400

        # Classify image
        img_result = classify_image(image)
        if isinstance(img_result, dict) and "error" in img_result:
            return jsonify({"error": img_result["error"]}), 400
            
        img_risk = img_result if isinstance(img_result, int) else 0

        # Calculate composite risk score
        score = img_risk
        score += 1 if weather['humidity'] < 30 else 0
        score += 1 if weather['temp'] > 30 else 0
        score += 1 if weather['wind'] > 10 else 0

        # Determine risk level
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
                "image_risk_score": img_risk
            }
        })

    except Exception as e:
        logger.error(f"Unexpected error in risk assessment: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)