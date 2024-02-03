import base64
from io import BytesIO
import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
import PyPDF2

# from dotenv import load_dotenv

# load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
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


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


@st.cache_data
def get_doc_info(uploaded_file):
    response = get_model_response(
        uploaded_file,
        user_prompt="Suggest 3 questions to ask about this document",
    )

    markdown = None
    image = None
    if uploaded_file.type == "application/pdf":
        # text = extract_text_from_pdf(uploaded_file=uploaded_file)
        # Convert the file to a data URL
        b64 = base64.b64encode(uploaded_file.getvalue()).decode()
        pdf_url = f"data:application/pdf;base64,{b64}"
        # Create an iframe with the data URL as the source
        # markdown = f'<iframe src="{pdf_url}" width="700" height="700"></iframe>'
        markdown = (
            f'<embed src="{pdf_url}" type="application/pdf" width="700" height="700" />'
        )

        # st.download_button("Download PDF", upload_image_file.getvalue(), file_name=upload_image_file.name)
    else:
        image = Image.open(uploaded_file)

    return response, markdown, image


def main():
    # Initialize the Streamlit App
    st.set_page_config(page_title="TalkinDocs", layout="wide")
    init_session_state()

    st.title("TalkinDocs")

    col1, col2 = st.columns([1, 1])
    with col2:
        uploaded_doc_file = st.file_uploader(
            "Choose an Image or PDF of the document",
            type=["jpg", "jpeg", "png", "pdf"],
        )

    with col1:

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

        user_prompt = st.chat_input(
            "What would you like to know about this document?",
            disabled=uploaded_doc_file is None,
        )
        # st.text_input(
        #     "What would you like to know about this document?",
        #     key="user_prompt_input_text",
        #     on_change=capture_user_prompt,
        # )
        # submit = st.button("Ask your question")

    if uploaded_doc_file and user_prompt:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_prompt, "ref_file": uploaded_doc_file.name}
        )
        with col1:
            with st.spinner("Thinking..."):
                response = get_model_response(uploaded_doc_file, user_prompt)
                st.session_state.chat_history.append(
                    {
                        "role": "model",
                        "content": response,
                        "ref_file": uploaded_doc_file.name,
                    }
                )

    if uploaded_doc_file is not None:
        with st.spinner("Processing..."):
            with col2:
                res, markdown, image = get_doc_info(uploaded_doc_file)

                st.subheader("Question Ideas")
                st.write(res)
                if markdown:
                    st.markdown(markdown, unsafe_allow_html=True)
                if image:
                    st.image(image, caption="Uploaded Image", use_column_width=True)

    with col1:
        for chat in st.session_state.chat_history:
            is_model_msg = chat["role"] == "model"
            with st.chat_message("AI" if is_model_msg else "User"):
                author = "DocMaster" if is_model_msg else "You"
                st.write(f"**{author}**")
                st.write(chat["content"])
                st.caption(f"Document: {chat['ref_file']}")


if __name__ == "__main__":
    main()
