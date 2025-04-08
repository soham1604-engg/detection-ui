import streamlit as st
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocess_data(df):
    # Process AI-generated text
    df_ai = df[['AI Text 1', 'AI Text 2', 'AI Text 3']].stack().reset_index(level=1, drop=True).to_frame(name='text')
    df_ai['generated'] = 1  # Label AI text as 1
    
    # Process Human-written text
    df_human = df[['Human Written Text']].dropna().rename(columns={'Human Written Text': 'text'})
    df_human['generated'] = 0  # Label human-written text as 0
    
    # Concatenate AI and human-written text into one dataframe
    df_processed = pd.concat([df_ai, df_human], ignore_index=True)
    
    # Balance the dataset
    min_samples = min(len(df_ai), len(df_human))
    df_ai = df_ai.sample(min_samples, random_state=42)
    df_human = df_human.sample(min_samples, random_state=42)
    df_balanced = pd.concat([df_ai, df_human], ignore_index=True)
    
    return df_balanced

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def train_model(df):
    df['text'] = df['text'].apply(clean_text)
    
    X = df['text']
    y = df['generated']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = SVC()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    joblib.dump((model, vectorizer), "model.pkl")
    
    return accuracy

def predict_text(text):
    try:
        model, vectorizer = joblib.load("model.pkl")
        text_cleaned = clean_text(text)
        text_tfidf = vectorizer.transform([text_cleaned])
        prediction = model.predict(text_tfidf)[0]
        return "AI-Generated" if prediction == 1 else "Human-Written"
    except Exception as e:
        return f"Error: {str(e)}"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Preprocessing & Training", "Detector"])

if page == "Preprocessing & Training":
    st.title("AI Text Detection - Preprocessing and Training")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("### Original Data:", df.head())
        
        processed_df = preprocess_data(df)
        st.write("### Processed Data:", processed_df.head())
        
        # Save processed file
        processed_file = "processed_dataset.xlsx"
        processed_df.to_excel(processed_file, index=False)
        
        with open(processed_file, "rb") as f:
            st.download_button("Download Processed File", f, file_name="processed_dataset.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # Train Model
        accuracy = train_model(processed_df)
        st.write(f"### Model Trained Successfully! Accuracy: {accuracy:.2f}")

elif page == "Detector":
    st.title("AI Text Detector")
    input_text = st.text_area("Enter text to classify:")
    if st.button("Detect"):
        result = predict_text(input_text)
        st.write(f"### Prediction: {result}")