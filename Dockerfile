FROM python:3.11-slim

WORKDIR /app

# Install dependencies first(cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files and data
COPY . .

# Default container entry point for Render/Cloud deployment
# Uses lightweight TF-IDF version (streamlit_app_tfidf.py) to run within memory limits
# Switch to streamlit_app.py if running locally or on paid instances with 1GB+ RAM
CMD ["streamlit", "run", "streamlit_app_tfidf.py", "--server.port=10000", "--server.address=0.0.0.0"]
