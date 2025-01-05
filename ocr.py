import streamlit as st
import tempfile
from PIL import Image
import pytesseract
import openai
import google.generativeai as genai

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\OWNER\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

# Define AI interpretation functions


def interpret_with_openai(ocr_text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Interpret the following text: {ocr_text}",
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def interpret_with_gemini(ocr_output, gemini_api_key):
    """Interprets OCR output using Google Gemini.

    Args:
        ocr_output: The OCR output to interpret.
        gemini_api_key: Your Gemini API key.

    Returns:
        The Gemini model's response.
    """
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            "You are a helpful assistant interpreting OCR output: " + ocr_output)
        return response.text
    except Exception as e:
        st.error(f"Error during Gemini interpretation: {e}")
        return None


# Main Streamlit app logic
st.title("OCR and Generative AI Note Interpreter")

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image of handwritten notes", type=["jpg", "png"])

if uploaded_file is not None:
    with st.spinner("Processing image..."):
        try:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                image_path = temp_file.name

            # Perform OCR using pytesseract
            try:
                image = Image.open(image_path)
                ocr_output = pytesseract.image_to_string(image)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                ocr_output = None

            if not ocr_output:
                st.error("OCR failed to extract text. Please try another image.")
            else:
                st.write("**OCR Output:**")
                st.write(ocr_output)

                # Select AI model
                model_choice = st.selectbox(
                    "Choose an AI model:", ["OpenAI", "Gemini"])

                if model_choice == "OpenAI":
                    openai_api_key = st.text_input(
                        "Enter your OpenAI API key:", type="password")
                    if openai_api_key:
                        openai.api_key = openai_api_key
                        try:
                            openai_response = interpret_with_openai(ocr_output)
                            if openai_response:
                                st.write("**OpenAI Interpretation:**")
                                st.write(openai_response)
                            else:
                                st.warning(
                                    "No response from OpenAI. Please check your input or try again.")
                        except Exception as e:
                            st.error(f"Error with OpenAI interpretation: {e}")

                elif model_choice == "Gemini":
                    gemini_api_key = st.text_input(
                        "Enter your Gemini API key:", type="password")
                    if gemini_api_key:
                        try:
                            gemini_response = interpret_with_gemini(
                                ocr_output, gemini_api_key)
                            if gemini_response:
                                st.write("**Gemini Interpretation:**")
                                st.write(gemini_response)
                            else:
                                st.warning(
                                    "No response from Gemini. Please check your input or try again.")
                        except Exception as e:
                            st.error(f"Error with Gemini interpretation: {e}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Add export option
if "ocr_output" in locals() and ocr_output:
    if st.button("Download OCR Results"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_text_file:
                temp_text_file.write(ocr_output.encode("utf-8"))
                st.success(f"OCR results saved to {temp_text_file.name}")
        except Exception as e:
            st.error(f"Error saving OCR results: {e}")
