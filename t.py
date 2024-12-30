import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Streamlit app
st.title("Exhaustion Score Predictor with Dashboard and Analysis")


@st.cache_data
def preprocess_text(text):
    # Replace with the real preprocessing logic
    return text.lower().strip()


@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    data = data.dropna(subset=['processed_comment'])
    return data


@st.cache_resource
def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train_vec, y_train)

    # Predictions and Metrics
    y_pred = model.predict(X_test_vec)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, vectorizer, mse, r2, y_test, y_pred


# Manually load the dataset
try:
    filepath = "labbled.csv"  # Update the path if necessary
    data = load_data(filepath)
    st.write("Dataset loaded successfully!")
    st.write(data.head())
except FileNotFoundError:
    st.error(f"File not found: {filepath}")
    st.stop()

# Dataset Overview
st.sidebar.header("Dataset Overview")
st.sidebar.write("Number of rows:", len(data))
st.sidebar.write("Number of columns:", len(data.columns))

# Analysis Section
st.header("Dataset Analysis")
if st.checkbox("Show Dataset Summary"):
    st.write(data.describe())

if st.checkbox("Show Exhaustion Score Distribution"):
    st.write("Distribution of Exhaustion Scores")
    fig, ax = plt.subplots()
    sns.histplot(data['exhaustion_score'], kde=True, ax=ax, color="blue")
    st.pyplot(fig)

# Prepare data for training
X = data['processed_comment']
y = data['exhaustion_score']
model, vectorizer, mse, r2, y_test, y_pred = train_model(X, y)

# Display KPIs
st.header("Key Performance Indicators (KPIs)")
st.metric("Mean Squared Error", f"{mse:.2f}")
st.metric("R-squared", f"{r2:.2f}")

# Dashboard
st.header("Dashboard")
st.subheader("Actual vs Predicted Exhaustion Scores")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5, color="orange")
ax.plot([y_test.min(), y_test.max()], [y_test.min(),
        y_test.max()], color="blue", linestyle="--")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

st.subheader("Feature Importance")
if st.checkbox("Show Feature Importance (Top 20 TF-IDF Features)"):
    feature_array = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_
    sorted_indices = np.argsort(-coefficients)
    top_features = feature_array[sorted_indices[:20]]
    top_coefficients = coefficients[sorted_indices[:20]]

    importance_df = pd.DataFrame({
        "Feature": top_features,
        "Coefficient": top_coefficients
    }).sort_values(by="Coefficient", ascending=False)

    st.write(importance_df)
    fig, ax = plt.subplots()
    sns.barplot(data=importance_df, x="Coefficient",
                y="Feature", ax=ax, palette="viridis")
    st.pyplot(fig)

# Prediction Section
st.header("Predict Exhaustion Score")
input_text = st.text_area("Enter text to predict exhaustion score:")
if input_text:
    processed_text = preprocess_text(input_text)
    text_vec = vectorizer.transform([processed_text])
    predicted_score = model.predict(text_vec)[0]
    st.write(f"Predicted Exhaustion Score: {predicted_score:.2f}")
