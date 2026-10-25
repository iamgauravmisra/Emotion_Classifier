import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# Helper functions matching the notebook logic
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = "".join([i for i in text if not i.isdigit()])
    # Remove non-ascii characters (emojis, etc.)
    text = "".join([i for i in text if i.isascii()])
    # Tokenize and remove stopwords
    words = word_tokenize(text)
    cleaned = [i for i in words if i not in stop_words]
    return " ".join(cleaned)

# Load saved models and mappings
@st.cache_resource
def load_artifacts():
    model = joblib.load('logistic_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    mapping = joblib.load('emotion_mapping.pkl')
    return model, vectorizer, mapping

try:
    model, vectorizer, mapping = load_artifacts()
    st.title("🎭 Emotion Classifier App")
    st.write("Type a sentence below to predict the underlying emotion.")

    # User Input
    user_input = st.text_area("Enter Text:", "I am so happy that everything worked out!")

    if st.button("Predict"):
        if user_input.strip() != "":
            # 1. Clean
            cleaned_input = clean_text(user_input)
            # 2. Vectorize
            vectorized_input = vectorizer.transform([cleaned_input])
            # 3. Predict
            prediction = model.predict(vectorized_input)[0]
            predicted_emotion = mapping.get(prediction, "Unknown")
            
            # Get probability scores
            probabilities = model.predict_proba(vectorized_input)[0]
            
            st.success(f"Predicted Emotion: **{predicted_emotion.capitalize()}**")
            
            # Show all probabilities
            st.write("### Confidence Scores:")
            for idx, prob in enumerate(probabilities):
                emotion_name = mapping.get(idx, f"Class {idx}")
                st.progress(float(prob))
                st.write(f"{emotion_name.capitalize()}: {prob*100:.2f}%")
        else:
            st.warning("Please enter some text.")
except Exception as e:
    st.error(f"Error loading artifacts: {e}\nMake sure the .pkl files are in the same folder.")
