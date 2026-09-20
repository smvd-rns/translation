FROM python:3.11-slim

# Install ffmpeg for audio chunking only (no heavy model loading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Patch default Streamlit HTML title so it shows Translation on initial page load
RUN python -c "import streamlit, os; p = os.path.join(os.path.dirname(streamlit.__file__), 'static', 'index.html'); html = open(p, 'r').read().replace('<title>Streamlit</title>', '<title>Translation - Audio & Video Transcription</title>'); open(p, 'w').write(html)"

COPY . .

EXPOSE 8501

CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
