import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


DB_FAISS_PATH="vectorstore/db_faiss"

#pAGE CONFIGURATION
st.set_page_config(
    page_title="Medibot - Your Medical Assistant",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .source-docs {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin-top: 15px;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        color: #856404;
    }
    
    .feature-box {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm


def main():
    st.markdown('<h1 class="main-header">&#127973; Medibot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your AI-Powered Medical Assistant</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## &#8505;&#65039; About Medibot")
        st.markdown("""
        <div class="feature-box">
        <strong>Features:</strong><br>
        • Medical knowledge base queries<br>
        • Source document references<br>
        • Accurate, context-based answers<br>
        • Safe and reliable responses
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## &#127919; Tips for Better Results")
        st.markdown("""
        - Be specific with your medical questions
        - Ask about symptoms, treatments, or conditions
        - Review source documents for verification
        """)
        
        st.markdown("## &#9881;&#65039; System Status")
        try:
            vectorstore = get_vectorstore()
            if vectorstore:
                st.success("\u2705 Knowledge base loaded")
            else:
                st.error("\u274C Knowledge base error")
        except Exception as e:
            st.error(f"\u274C System error: {str(e)}")
    #DISCLAIMER
    st.markdown("""
    <div class="warning-box">
    <strong>&#9888;&#65039; Medical Disclaimer:</strong> This chatbot provides general medical information for educational purposes only. 
    Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)

    # Chat interface
    st.markdown("## \U0001F4AC Chat with Medibot")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                if message['role'] == 'assistant' and 'Source Docs:' in message['content']:
                    # SPLITTING CONTENT AND SOURCE DOCUMENTS
                    parts = message['content'].split('\nSource Docs:\n')
                    st.markdown(parts[0])
                    if len(parts) > 1:
                        st.markdown(f"""
                        <div class="source-docs">
                        <strong>&#128218; Source Documents:</strong><br>
                        {parts[1]}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(message['content'])

    prompt = st.chat_input("Ask me about medical conditions, symptoms, treatments, or general health questions...")

    if prompt:
        st.session_state.messages.append({'role':'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        # Generate response
        with st.chat_message('assistant'):
            with st.spinner('\U0001F50D Searching medical knowledge base...'):
                CUSTOM_PROMPT_TEMPLATE = """
                        Use the pieces of information provided in the context to answer user's question.
                        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                        Dont provide anything out of the given context

                        Context: {context}
                        Question: {question}

                        Start the answer directly. No small talk please.
                        """
                
                try: 
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        st.error("\u274C Failed to load the vector store")
                        return

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatGroq(
                            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  
                            temperature=0.0,
                            groq_api_key=os.environ["GROQ_API_KEY"],
                        ),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                    )

                    response = qa_chain.invoke({'query':prompt})

                    result = response["result"]
                    source_documents = response["source_documents"]
                    
                    # DISPLAYING RESULT
                    st.markdown(result)
                    
                    # DISPLAYING SOURCE DOCUMENTS
                    if source_documents:
                        st.markdown(f"""
                        <div class="source-docs">
                        <strong>&#128218; Source Documents:</strong><br>
                        {str(source_documents)}
                        </div>
                        """, unsafe_allow_html=True)
        
                    result_to_show = result + "\nSource Docs:\n" + str(source_documents)
                    st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

                except Exception as e:
                    st.error(f"\u274C Error: {str(e)}")
                    st.info("Please try rephrasing your question or contact support if the issue persists.")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>Medibot v1.0</strong> | Powered by AI | Always consult healthcare professionals
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()