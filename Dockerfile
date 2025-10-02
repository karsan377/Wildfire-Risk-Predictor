# Use a lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port for Fly
EXPOSE 8080

# Run your app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "wilddd:app"]
