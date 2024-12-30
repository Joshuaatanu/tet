import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib

# Load model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@st.cache
def preprocess_text(text):
    return text.lower().strip()


# Header
st.title("Exhaustion Score Prediction Dashboard")
st.markdown("### Monitor data, model performance, and predictions")

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["Data Analysis", "Model Performance", "Live Predictions"])

with tab1:
    st.subheader("Data Analysis")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Overview:")
        st.dataframe(df.head())

        st.write("Exhaustion Score Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['exhaustion_score'], kde=True, bins=20)
        st.pyplot(plt)

        st.write("Word Cloud")
        text = " ".join(df['processed_comment'].dropna())
        wordcloud = WordCloud(width=800, height=400).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

with tab2:
    st.subheader("Model Performance")
    if uploaded_file is not None:
        X = df['processed_comment']
        y = df['exhaustion_score']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")

        st.write("Actual vs. Predicted")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual Scores")
        plt.ylabel("Predicted Scores")
        st.pyplot(plt)

with tab3:
    st.subheader("Live Predictions")
    input_text = st.text_area("Enter text for prediction:")
    if input_text:
        processed_text = preprocess_text(input_text)
        text_vec = vectorizer.transform([processed_text])
        predicted_score = model.predict(text_vec)[0]
        st.write(f"Predicted Exhaustion Score: {predicted_score:.2f}")
