import base64
from io import BytesIO
import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
assert len(GOOGLE_API_KEY) > 0, "Please set the GOOGLE_API_KEY in the .env file"
genai.configure(api_key=GOOGLE_API_KEY)


def input_image_bytes(uploaded_file):
    if uploaded_file is not None:
        # Convert the Uploaded File into bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded")


def extract_text_from_pdf(uploaded_file):
    uploaded_file = BytesIO(uploaded_file.getvalue())
    # Create a PdfFileReader object
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page_obj: PyPDF2.PageObject = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    return text


# Load the appropriate Gemini model based on the file type
def load_gemini_model(file_type):
    if file_type == "application/pdf":
        return genai.GenerativeModel("models/gemini-pro")
    else:
        return genai.GenerativeModel("models/gemini-pro-vision")


# Get response from Gemini based on the model
def get_gemini_response(
    model: genai.GenerativeModel, input_prompt, content, user_input_prompt
):
    if isinstance(content, str):  # Assuming text content for PDF
        response = model.generate_content([input_prompt, content, user_input_prompt])
    else:  # Assuming image content
        response = model.generate_content([input_prompt, content[0], user_input_prompt])
    return response.parts[0].text


input_prompt = """
You are an expert in understanding all sorts of documents such as invocies, forms, 
financials, images, etc. Please try to answer the question using the information 
from the uploaded document.
"""


def get_model_response(upload_image_file, user_prompt):
    model = load_gemini_model(upload_image_file.type)
    if upload_image_file.type == "application/pdf":
        input_content = extract_text_from_pdf(
            upload_image_file
        )  # Assuming this returns the text as a string
    else:
        input_content = input_image_bytes(upload_image_file)  # Image data
    response = get_gemini_response(model, input_prompt, input_content, user_prompt)
    return response


def capture_user_prompt():
    st.session_state.user_prompt = st.session_state.user_prompt_input_text
    st.session_state.user_prompt_input_text = ""


def init_session_state():
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""


def main():
    # Initialize the Streamlit App
    st.set_page_config(page_title="TalkinDocs")
    init_session_state()

    uploaded_doc_file = st.file_uploader(
        "Choose an Image or PDF of the document",
        type=["jpg", "jpeg", "png", "pdf"],
    )

    with st.sidebar:
        st.title("TalkinDocs")
        st.write(
            "This app uses the Google AI Generative Model to answer questions about documents."
        )
        st.write(
            "Upload an image or a PDF of the document and ask questions to get answers."
        )
        st.write(
            "The model will also suggest questions to ask about the document if you don't have any."
        )
        st.divider()
        st.text_input(
            "What would you like to know about this document?",
            key="user_prompt_input_text",
            on_change=capture_user_prompt,
        )
        submit = st.button("Ask your question")

    user_prompt = st.session_state.user_prompt
    st.session_state.user_prompt = ""

    print("uploaded_doc_file", uploaded_doc_file)

    if uploaded_doc_file and user_prompt:
        with st.spinner("Processing..."):
            response = get_model_response(uploaded_doc_file, user_prompt)
            st.subheader("Answer")
            st.write(response)

    elif uploaded_doc_file is not None:
        response = get_model_response(
            uploaded_doc_file,
            user_prompt="Suggest 3 questions to ask about this document",
        )
        st.subheader("Example Questions")
        st.write(response)

        if uploaded_doc_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file=uploaded_doc_file)
            # Convert the file to a data URL
            b64 = base64.b64encode(uploaded_doc_file.getvalue()).decode()
            pdf_url = f"data:application/pdf;base64,{b64}"
            # Create an iframe with the data URL as the source
            st.markdown(
                f'<iframe src="{pdf_url}" width="700" height="700"></iframe>',
                unsafe_allow_html=True,
            )
            st.write(text)
            # st.download_button("Download PDF", upload_image_file.getvalue(), file_name=upload_image_file.name)
        else:
            image = Image.open(uploaded_doc_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)


if __name__ == "__main__":
    main()
