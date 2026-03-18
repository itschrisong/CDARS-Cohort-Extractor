FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Intro.py .
COPY core/ core/
COPY pages/ pages/
COPY configs/ configs/

# Create data directories
RUN mkdir -p data/raw data/processed

# Disable Streamlit telemetry and browser auto-open
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "Intro.py", "--server.port=8501", "--server.address=0.0.0.0"]
