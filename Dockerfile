FROM python:3.10-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Kodları kopyala
COPY src/ ./src/

# Model önbelleğini indir (isteğe bağlı, ilk çalıştırmada da inebilir)
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='axilion/RoBERTa-movie-sentiment-analyzer')"

# Başlangıç scripti ekle
COPY start.sh .

# Portları aç
EXPOSE 8000 8501

# Başlatıcı scripti çalıştır
CMD ["bash", "start.sh"]