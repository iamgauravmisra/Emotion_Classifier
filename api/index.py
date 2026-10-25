from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Initialize FastAPI app
app = FastAPI(title="Emotion Classifier API")

# Ensure NLTK resources are available locally
nltk_data_path = os.path.join(os.path.dirname(__file__), "../nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)
    nltk.download("punkt", download_dir=nltk_data_path)

stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join([i for i in text if not i.isdigit()])
    text = "".join([i for i in text if i.isascii()])
    words = word_tokenize(text)
    cleaned = [i for i in words if i not in stop_words]
    return " ".join(cleaned)

# Load artifacts
# Note: artifacts should be placed in the root directory
model_path = os.path.join(os.path.dirname(__file__), "../logistic_model.pkl")
tfidf_path = os.path.join(os.path.dirname(__file__), "../tfidf_vectorizer.pkl")
mapping_path = os.path.join(os.path.dirname(__file__), "../emotion_mapping.pkl")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(tfidf_path)
    mapping = joblib.load(mapping_path)
except Exception as e:
    model, vectorizer, mapping = None, None, None
    print(f"Error loading models: {e}")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "alive", "message": "Emotion Classifier API is running. POST to /predict"}

@app.post("/predict")
def predict(data: TextInput):
    if not model or not vectorizer or not mapping:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded on server.")
    
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
        
    cleaned = clean_text(data.text)
    vectorized = vectorizer.transform([cleaned])
    
    # Get predictions
    pred_idx = int(model.predict(vectorized)[0])
    emotion = mapping.get(pred_idx, "unknown")
    
    # Get probabilities
    probs = model.predict_proba(vectorized)[0]
    confidence_scores = {mapping.get(i, f"class_{i}"): float(prob) for i, prob in enumerate(probs)}
    
    return {
        "input": data.text,
        "prediction": emotion,
        "confidence": confidence_scores
    }
