FROM python:3.10-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='axilion/RoBERTa-movie-sentiment-analyzer')"

COPY start.sh .

EXPOSE 8000 8501

CMD ["bash", "start.sh"]