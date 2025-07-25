# Camp VeriFIRE: Image + Weather Based Fire Risk Prediction

## Overview
Camp VeriFIRE is a lightweight web service that predicts wildfire risk at a specific location by combining:
- A convolutional neural network (CNN) trained on outdoor imagery
- Real-time weather data (temperature, humidity, wind speed) from OpenWeatherMap
- A Flask-based API with a simple HTML/JS frontend

## Features
- Upload a photo of your surroundings (campground, backcountry site, etc.) and get a wildfire risk score
- Combines CNN predictions with weather multipliers to account for temperature, wind, and humidity
- Flask API endpoints for developers, plus a lightweight web interface for end users
- Pretrained MobileNetV2-based model for fast inference

## Installation
### Prerequisites
- Python 3.9+
- OpenWeatherMap API key (free at [openweathermap.org](https://openweathermap.org/))

### Setup
```bash
git clone https://github.com/karsan377/Wildfire-Risk-Predictor.git
cd Wildfire-Risk-Predictor

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
# OR
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variable
echo "OPENWEATHER_API_KEY=your_api_key_here" > .env

##
### Set up virtual environment
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "OPENWEATHER_API_KEY=your_api_key_here" > .env

```
## Usage

```bash
python wilddd.py
```

## API Endpoints

### POST /assess-risk
Classifies an image with weather data

Parameters:
- city or latitude and longitude
- image

```
Response JSON:
{
  "fire_risk": "High",
  "details": {
    "city": "San Diego",
    "temperature": 28.3,
    "humidity": 35,
    "wind_speed": 6.5,
    "image_risk_score": 0.721,
    "final_risk_score": 1.346
  }
}
```

### POST /verify-city
Checks city validity

```
Request (JSON):
{ "city": "San Diego" }
Response (JSON):
{ "message": "City verified successfully" }
```

### Prerequisites
- Python 3.9+
- OpenWeatherMap API key (free at [openweathermap.org](https://openweathermap.org/))

