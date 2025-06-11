# 🎬 RoBERTa-Based Sentiment Analysis on IMDB Reviews

- This project fine-tunes a pre-trained RoBERTa model to classify IMDB movie reviews as **positive** or **negative**.
- It includes data preprocessing, model training, serving with FastAPI, a user interface built with Streamlit, and containerization with Docker.
- The model is also hosted on the Hugging Face Hub.

---

## 📂 Project Structure

```
ROBERTA-SENTIMENT-ANALYSIS/
│
├── data/
│   ├── raw/
│   │   └── IMDB Dataset.csv         # Original dataset.
│   └── preprocessed/
│       └── Cleaned_IMDB_Dataset.csv # Cleaned dataset.
│
├── models/                          # Trained model weights (not in repo)
│   └── model_weights.pth
│
├── notebooks/
│   ├── data_preprocessing.ipynb     # EDA and data cleaning
│   └── model_training.ipynb         # RoBERTa fine-tuning and evaluation
│
├── src/
│   ├── app.py                       # Model loading and prediction logic
│   ├── fast_api.py                  # FastAPI backend (REST API)
│   └── streamlit_app.py             # Streamlit interface
│
├── .gitignore
├── .dockerignore
├── Dockerfile
├── start.sh
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/axilion19/RoBERTa-sentiment-analysis.git
cd RoBERTa-sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch FastAPI

```bash
uvicorn src.fast_api:app --reload
```

- API: [http://localhost:8000](http://localhost:8000)

### 4. Launch Streamlit Interface

```bash
streamlit run src/streamlit_app.py
```

- Interface: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Run with Docker

1. Build the Docker image:

   ```bash
   docker build -t movie-sentiment-app .
   ```

2. Run the container:

   ```bash
   docker run -p 8000:8000 -p 8501:8501 movie-sentiment-app
   ```

- FastAPI: [http://localhost:8000](http://localhost:8000)
- Streamlit: [http://localhost:8501](http://localhost:8501)

---

## 🤗 Hugging Face Model

The model is hosted on Hugging Face Hub:  
[axilion/RoBERTa-movie-sentiment-analyzer](https://huggingface.co/axilion/RoBERTa-movie-sentiment-analyzer)

You can easily use it in your own code:

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="axilion/RoBERTa-movie-sentiment-analyzer")
print(classifier("The movie was surprisingly touching and well-acted."))
```

---
