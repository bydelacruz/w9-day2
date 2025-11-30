# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
# We copy requirements first to leverage Docker cache layers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY models ./models
# Copy data if you want it inside the container, 
COPY data ./data 

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# Host 0.0.0.0 is crucial for Docker networking
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
