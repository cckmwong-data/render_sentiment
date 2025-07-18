# Use official python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
apt-get install -y gcc build-essential git wget && \
rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy app code
COPY app.py ./
COPY max_length_sentiment.txt ./
COPY tokenizer_sentiment.json ./

# Copy requirements first (for better caching)
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download model from huggingface
RUN wget -O sentiment_lstm_model.h5 "https://huggingface.co/cckmwong/sentiment/resolve/main/sentiment_lstm_model.h5"

# Download NLTK data at build time to a fixed directory
RUN python -m nltk.downloader -d /app/nltk_data punkt averaged_perceptron_tagger stopwords wordnet

# Set environment variable for NLTK
ENV NLTK_DATA="/app/nltk_data"

# Set Streamlit to listen on all interfaces (important for docker)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 8501

# Launch app
CMD ["streamlit", "run", "app.py"]
