# Use an official Python runtime as the base image
FROM python:3.11-slim

# Install system dependencies required for OpenCV (cv2) and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables (if needed, e.g., Flask)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production  # Use 'development' for debugging

# Expose the port Flask runs on (default: 5000)
EXPOSE 5000

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]