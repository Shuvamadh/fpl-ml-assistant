# Hugging Face Spaces (Docker SDK) deployment for the Streamlit app.
# Pinning everything here means the Python version and dependency set can
# never silently drift on a redeploy the way it did on Streamlit Community
# Cloud, where the Python version was only ever a dashboard setting.
FROM python:3.11-slim

# lightgbm's wheel needs libgomp at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY streamlit_app/requirements.txt streamlit_app/requirements.txt
RUN pip install --no-cache-dir -r streamlit_app/requirements.txt

COPY src/ src/
COPY streamlit_app/ streamlit_app/
COPY gui/league_extras.py gui/league_extras.py
COPY assets/ assets/
COPY models/ models/
COPY data/ data/

# Spaces' Docker SDK expects the app to listen on 7860.
EXPOSE 7860
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "streamlit_app/app.py"]
