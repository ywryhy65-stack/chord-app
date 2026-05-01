# Use python:3.10-slim as the base image
FROM python:3.10-slim

# Install system dependencies: ffmpeg, libsndfile1, and nodejs (for yt-dlp JS challenges)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U yt-dlp

# Copy the rest of the project files
COPY . .

# Expose port 8000
EXPOSE 8000

# Set the command to run the FastAPI app using uvicorn
# Based on the file name main.py, the module is 'main'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
