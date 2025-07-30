import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# ---- ENV + Config ----
os.environ["TRANSFORMERS_NO_GPU"] = "1"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
load_dotenv()

def message_bubble(text, is_user=True):
    bubble_style = """
        <div style="
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 10px;
            background-color: {bg_color};
            color: {text_color};
            max-width: 80%;
            align-self: {alignment};
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            font-family: Arial, sans-serif;
            font-size: 16px;">
            {text}
        </div>
    """

    return bubble_style.format(
        text=text,
        bg_color="#f0f0f0" if is_user else "#007bff",
        text_color="#000" if is_user else "#fff",
        alignment="flex-end" if is_user else "flex-start",
    )

st.set_page_config(page_title="RAG ChatBot", layout="wide")
st.title("RAG ChatBot💬")

# ---- Session Setup ----
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ---- File Upload ----
uploaded_pdf = st.file_uploader("Upload your PDF here", type="pdf")

@st.cache_resource(show_spinner=False)
def get_vectorstore(pdf_path):
    loaders = [PyPDFLoader(pdf_path)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='./all-MiniLM-L12-v2', model_kwargs={"device": "cpu"}),

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

# ---- LLM Setup ----
groq_chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

system_prompt = ChatPromptTemplate.from_template(
    """You are very smart at everything, you always give the best, 
    the most accurate and most precise answers. Answer the following Question: {user_prompt}.
    Start the answer directly. No small talk please"""
)

# ---- Show past messages ----
def message_bubble(content, is_user=True):
    background = "#2d8535" if is_user else "#006bb3"  # Blue for user, dark gray for assistant
    text_color = "#fff" if is_user else "#fff"
    align = "flex-end" if is_user else "flex-start"
    border_radius = "20px 20px 5px 20px" if is_user else "20px 20px 20px 5px"

    return f"""
    <div style="
        display: flex;
        justify-content: {align};
        margin-bottom: 10px;
    ">
        <div style="
            background-color: {background};
            color: {text_color};
            padding: 12px 18px;
            border-radius: {border_radius};
            max-width: 75%;
            word-wrap: break-word;
            font-size: 15px;
            box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.3);
        ">
            {content}
        </div>
    </div>
    """


st.markdown('<div style="display: flex; flex-direction: column;">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    st.markdown(message_bubble(msg["content"], is_user=(msg["role"] == "user")), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

   

# ---- Chat Input ----
prompt = st.chat_input("What can I help you with?")

if prompt:
    st.markdown(message_bubble(prompt, is_user=True), unsafe_allow_html=True)

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        # If PDF is uploaded, use RAG
        if uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                tmp_path = tmp_file.name

            with st.spinner("Indexing your PDF..."):
                vectorstore = get_vectorstore(tmp_path)

            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            result = chain({"query": prompt})
            response = result["result"]

        else:
            # No PDF uploaded – use LLM directly
            formatted_prompt = system_prompt.invoke({"user_prompt": prompt})
            response = groq_chat.invoke(formatted_prompt).content

        st.markdown(message_bubble(response, is_user=False), unsafe_allow_html=True)

        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
