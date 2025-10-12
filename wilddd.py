from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from flask_cors import CORS
import requests
from PIL import Image
import tensorflow as tf
import numpy as np
import logging
import io
import os
import json
import time
import threading
from datetime import datetime, timedelta

load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Cache configuration
WEATHER_CACHE_FILE = 'weather_cache.json'
CACHE_DURATION_HOURS = 1  # How long to keep cached data

# Popular camping cities (you can expand this list)
POPULAR_CAMPING_CITIES = [
    "Aspen,CO,US", "Aspen",  # Add both formats
    "Bend,OR,US", "Bend",
    "Sedona,AZ,US", "Sedona", 
    "Gatlinburg,TN,US", "Gatlinburg",
    "Estes Park,CO,US", "Estes Park",
    "South Lake Tahoe,CA,US", "South Lake Tahoe",  # Both formats
    "Moab,UT,US", "Moab",
    "Bar Harbor,ME,US", "Bar Harbor",
    "Jackson,WY,US", "Jackson",  # Both formats
    "Asheville,NC,US", "Asheville",
    "Flagstaff,AZ,US", "Flagstaff",
    "Bozeman,MT,US", "Bozeman",
    "Breckenridge,CO,US", "Breckenridge",
    "St. George,UT,US", "St. George",
    "Helen,GA,US", "Helen",
    "Leavenworth,WA,US", "Leavenworth"
]

# Global cache dictionary
weather_cache = {}

# Load the custom wildfire model once
try:
    app.model = tf.keras.models.load_model('wildfire_transfer_model.h5')
    logger.info("Custom wildfire model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    app.model = None

# ------------------ Cache Management Functions ------------------
def load_cache():
    """Load cache from file"""
    global weather_cache
    try:
        if os.path.exists(WEATHER_CACHE_FILE):
            with open(WEATHER_CACHE_FILE, 'r') as f:
                weather_cache = json.load(f)
            logger.info(f"Loaded cache with {len(weather_cache)} entries")
        else:
            weather_cache = {}
            logger.info("No existing cache file found, starting fresh")
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        weather_cache = {}

def save_cache():
    """Save cache to file"""
    try:
        with open(WEATHER_CACHE_FILE, 'w') as f:
            json.dump(weather_cache, f, indent=2)
        logger.debug("Cache saved successfully")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def is_cache_valid(cache_timestamp):
    """Check if cache entry is still valid"""
    try:
        cache_time = datetime.fromisoformat(cache_timestamp)
        return datetime.now() - cache_time < timedelta(hours=CACHE_DURATION_HOURS)
    except:
        return False

def cleanup_cache():
    """Remove expired cache entries"""
    global weather_cache
    expired_count = 0
    valid_entries = {}
    
    for city, data in weather_cache.items():
        if is_cache_valid(data.get('timestamp', '')):
            valid_entries[city] = data
        else:
            expired_count += 1
    
    if expired_count > 0:
        logger.info(f"Cleaned up {expired_count} expired cache entries")
        weather_cache = valid_entries
        save_cache()

# ------------------ ETL Functions ------------------
def fetch_city_weather(city_name):
    """Fetch weather data for a single city"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        res = response.json()

        if 'main' not in res:
            logger.error(f"Unexpected weather API response for {city_name}: {res}")
            return None

        weather_data = {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "wind": res["wind"]["speed"],
            "city": res.get("name", city_name),
            "timestamp": datetime.now().isoformat()
        }
        
        return weather_data
    except Exception as e:
        logger.error(f"Weather API error for {city_name}: {str(e)}")
        return None

def run_etl_process():
    """Run ETL process to update cache for all popular camping cities"""
    logger.info("Starting ETL process for popular camping cities...")
    updated_count = 0
    
    for city in POPULAR_CAMPING_CITIES:
        try:
            # Check if we need to update this city
            cached_data = weather_cache.get(city)
            if cached_data and is_cache_valid(cached_data.get('timestamp', '')):
                continue  # Skip if cache is still valid
            
            # Fetch fresh data
            weather_data = fetch_city_weather(city)
            if weather_data:
                weather_cache[city] = weather_data
                updated_count += 1
                logger.debug(f"Updated weather data for {city}")
            
            # Small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing {city}: {e}")
            continue
    
    if updated_count > 0:
        save_cache()
        logger.info(f"ETL process completed. Updated {updated_count} cities")
    else:
        logger.info("ETL process completed. No updates needed")

def schedule_etl():
    """Schedule ETL process to run periodically"""
    while True:
        try:
            run_etl_process()
            # Run every hour
            time.sleep(12* 3600)
        except Exception as e:
            logger.error(f"ETL scheduler error: {e}")
            time.sleep(300)  # Wait 5 minutes before retrying

# ------------------ Utility Functions ------------------
def get_weather(lat=None, lon=None, city=None):
    """Fetch weather data from OpenWeather API by coordinates, city name, or cache"""
    try:
        # First, try to get from cache if city is provided
        if city:
            # Try exact match first
            if city in weather_cache:
                cached_data = weather_cache[city]
                if is_cache_valid(cached_data.get('timestamp', '')):
                    logger.debug(f"Using cached data for exact match: {city}")
                    return {
                        "temp": cached_data["temp"],
                        "humidity": cached_data["humidity"],
                        "wind": cached_data["wind"],
                        "city": cached_data["city"],
                        "source": "cache"
                    }
            
            # Try partial match (city name without state/country)
            for cached_city, cached_data in weather_cache.items():
                cached_city_name = cached_city.split(',')[0].lower().strip()
                input_city_name = city.split(',')[0].lower().strip()
                
                if cached_city_name == input_city_name and is_cache_valid(cached_data.get('timestamp', '')):
                    logger.debug(f"Using cached data for partial match: {cached_city} -> {city}")
                    return {
                        "temp": cached_data["temp"],
                        "humidity": cached_data["humidity"],
                        "wind": cached_data["wind"],
                        "city": cached_data["city"],
                        "source": "cache"
                    }
        
        # If not in cache or cache expired, try to fetch from API
        # But if we're likely offline, skip API and use cache even if expired
        if lat is not None and lon is not None:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                res = response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed, trying cache fallback: {e}")
                # If API fails and we have city, try cache even if expired
                if city:
                    return get_cache_fallback(city)
                raise e
                
        elif city:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                res = response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed for {city}, trying cache fallback: {e}")
                return get_cache_fallback(city)
        else:
            logger.error("No location provided for weather lookup")
            return {"error": "No location provided"}

        if 'main' not in res:
            logger.error(f"Unexpected weather API response: {res}")
            return {"error": "Invalid weather data received"}

        weather_data = {
            "temp": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "wind": res["wind"]["speed"],
            "city": res.get("name", city),
            "source": "api"
        }
        
        # Cache the result if it's for a city
        if city:
            weather_data["timestamp"] = datetime.now().isoformat()
            weather_cache[city] = weather_data
            save_cache()
        
        return weather_data
        
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
        # Final fallback - try cache even if expired
        if city:
            fallback_result = get_cache_fallback(city, allow_expired=True)
            if fallback_result:
                return fallback_result
        return {"error": f"Weather service unavailable: {str(e)}"}

def get_cache_fallback(city, allow_expired=False):
    """Try to find cached data for a city with flexible matching"""
    # Try exact match
    if city in weather_cache:
        cached_data = weather_cache[city]
        if allow_expired or is_cache_valid(cached_data.get('timestamp', '')):
            logger.info(f"Using cached data fallback for: {city}")
            return {
                "temp": cached_data["temp"],
                "humidity": cached_data["humidity"],
                "wind": cached_data["wind"],
                "city": cached_data["city"],
                "source": "cache_fallback"
            }
    
    # Try partial match (city name without state/country)
    input_city_name = city.split(',')[0].lower().strip()
    for cached_city, cached_data in weather_cache.items():
        cached_city_name = cached_city.split(',')[0].lower().strip()
        
        if cached_city_name == input_city_name:
            if allow_expired or is_cache_valid(cached_data.get('timestamp', '')):
                logger.info(f"Using cached data fallback (partial match): {cached_city} -> {city}")
                return {
                    "temp": cached_data["temp"],
                    "humidity": cached_data["humidity"],
                    "wind": cached_data["wind"],
                    "city": cached_data["city"],
                    "source": "cache_fallback"
                }
    
    logger.warning(f"No cached data found for: {city}")
    return None

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

# ------------------ API Endpoints ------------------
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
                "final_risk_score": round(final_risk_score, 3),
                "data_source": weather.get('source', 'api')
            }
        })

    except Exception as e:
        logger.error(f"Unexpected error in risk assessment: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/verify-city', methods=['POST'])
def verify_city():
    try:
        if request.is_json:
            data = request.get_json()
            city = data.get('city', '').strip()
        else:
            city = request.form.get('city', '').strip()

        if not city:
            return jsonify({'error': 'City name is required'}), 400

        weather = get_weather(city=city)
        if "error" in weather:
            return jsonify({"error": weather["error"]}), 400

        return jsonify({'message': f'City "{city}" verified successfully'})

    except Exception as e:
        logger.error(f"Error verifying city: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/cache/status', methods=['GET'])
def cache_status():
    """Get cache status and contents"""
    cleanup_cache()  # Clean up before showing status
    return jsonify({
        'total_cities': len(weather_cache),
        'popular_cities': POPULAR_CAMPING_CITIES,
        'cache_file': WEATHER_CACHE_FILE,
        'cached_cities': list(weather_cache.keys())
    })

@app.route('/cache/refresh', methods=['POST'])
def refresh_cache():
    """Manually trigger cache refresh"""
    try:
        run_etl_process()
        return jsonify({'message': 'Cache refresh completed successfully'})
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        return jsonify({'error': f'Cache refresh failed: {str(e)}'}), 500

# ------------------ Initialize and Run App ------------------
def initialize_app():
    """Initialize the application"""
    # Load existing cache
    load_cache()
    
    # Clean up expired entries
    cleanup_cache()
    
    # Run initial ETL process
    logger.info("Running initial ETL process...")
    run_etl_process()
    
    # Start ETL scheduler in background thread
    etl_thread = threading.Thread(target=schedule_etl, daemon=True)
    etl_thread.start()
    logger.info("ETL scheduler started")

if __name__ == '__main__':
    # Initialize cache and start ETL
    initialize_app()
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)