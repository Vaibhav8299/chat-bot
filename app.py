import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
import speech_recognition as sr

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_pdfs(pdf_docs):
    if not pdf_docs:
        return "No PDF files uploaded. Please upload at least one PDF file."

    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return "Processing complete. You can now ask questions."

def translate_text(text, target_language):
    if target_language == "en":
        return text
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = f"Translate the following text to {target_language}. Provide only the translated text without any additional comments:\n\n{text}"
    response = model.invoke(prompt)
    return response.content

def user_input(user_question, selected_language):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    translated_response = translate_text(response["output_text"], selected_language)
    return translated_response

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def main():
    st.set_page_config(page_title="AI-Powered PDF Analyzer", page_icon="📚")
    st.title("AI-Powered PDF Analyzer")
    st.markdown("**Upload your PDF files and get answers to your questions in multiple languages.**")

    tab1, tab2 = st.tabs(["Upload & Process", "Ask Questions"])

    with tab1:
        st.header("Step 1: Upload your PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF Files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                result = process_pdfs(pdf_docs)
            st.success(result)

    with tab2:
        st.header("Step 2: Ask a Question from the Processed PDFs")
        user_question = st.text_input("Ask a Question", placeholder="Type your question here...")
        language = st.selectbox("Select Language", ["en", "fr", "hi", "bn", "ta", "te", "mr"], index=0)

        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Generating response..."):
                    response = user_input(user_question, language)
                st.text_area("Reply", value=response, height=200)
                
                audio_file = text_to_speech(response, language)
                st.audio(audio_file, format='audio/mp3')
            else:
                st.warning("Please enter a question.")

    st.markdown("**Note:** Ensure that your microphone is working if you want to use voice input.")

if __name__ == "__main__":
    main()