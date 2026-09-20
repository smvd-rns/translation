FROM python:3.11-slim

# Install ffmpeg and required system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files first for layer caching
COPY requirements.txt .

# Install CPU version of PyTorch to keep image lightweight & fit memory limits
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy remaining project files
COPY . .

# Expose default port (Render will override via $PORT)
EXPOSE 8501

# Run Streamlit bound to 0.0.0.0 and dynamic $PORT for Render
CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
