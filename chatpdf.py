import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_classic.embeddings import HuggingFaceBgeEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

def get_pdf_text(pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader=PdfReader(pdf)
            for page in pdf_reader.pages:
                    text += page.extract_text()
        return text

def get_text_chunks(text):
        text_spiltter=CharacterTextSplitter(
                separator='\n',
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
                
        )
        chunks=text_spiltter.split_text(text)
        return chunks


def get_vectorstore(text_chunks):
        embeddings=HuggingFaceBgeEmbeddings()
        vector=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
        return vector
def get_conversation_chain(vectorstore):
    llm_model = Ollama(model="llama3")
    
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # History-aware retriever prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm_model, retriever, contextualize_q_prompt
    )
    
    # QA chain prompt
    qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \

{context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm_model, qa_prompt)
    
    # Create final RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke({
        "input": user_question,
        "chat_history": []
    })
    st.write(response["answer"])
    
    
st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")

if "conversation" not in st.session_state:
        st.session_state.conversation=None
    
    
st.header('Chat with multiple PDFs: books:')


user_question=st.chat_input("Ask a question about your documents:")
if user_question:
        handle_userinput(user_question)
with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your file",accept_multiple_files=True)
        if st.button("Process"):
                with st.spinner("Processing"):
                        raw_text=get_pdf_text(pdf_docs)
                        
                        text_chunks=get_text_chunks(raw_text)
                        
                        vectorstore=get_vectorstore(text_chunks)
                        
                        st.session_state.conversation=get_conversation_chain(vectorstore)
       
