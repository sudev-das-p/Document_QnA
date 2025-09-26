import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

# -------------------------------
# Initialization
# -------------------------------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if "analytics" not in st.session_state:
    st.session_state.analytics = {"questions": 0, "response_times": []}

if "config" not in st.session_state:
    st.session_state.config = {
        "llm_model": "Gemma2-9b-It",
        "system_prompt": """You are an assistant for question answering tasks. 
                            Use the provided context and conversation history. 
                            If the answer is unknown, say so.
                            Document Context:
                            {context}"""
    }

# -------------------------------
# Helper functions
# -------------------------------
def create_chain(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    persist_dir = "./chroma_db_new"
    # if os.path.exists(persist_dir): 
    #     import shutil
    #     shutil.rmtree(persist_dir)

    db = Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    llm = ChatGroq(groq_api_key=groq_key, model=st.session_state.config["llm_model"])

    # History aware retriever
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Rewrite the question to be standalone. Do not answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # QnA chain
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.config["system_prompt"]),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    qna_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state:
            st.session_state[session] = ChatMessageHistory()
        return st.session_state[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


# -------------------------------
# Pages
# -------------------------------
def user_page():
    st.title("📄 User Page")
    st.write("Upload your PDF and chat with it.")

    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        docs = []
        for file in uploaded_files:
            temp_path = "./temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(temp_path)
            docs.extend(loader.load())

        if "chat_chain" not in st.session_state:
            st.session_state.chat_chain = create_chain(docs)
            st.session_state.chat_history = []  # store chat messages

        chain = st.session_state.chat_chain
        # user_input = st.chat_input("💬 Ask your question")

        # if user_input:
        #     start = time.time()
        #     response = chain.invoke(
        #         {"input": user_input},
        #         config={"configurable": {"session_id": "user_session"}}
        #     )
        #     end = time.time()

        #     st.session_state.analytics["questions"] += 1
        #     st.session_state.analytics["response_times"].append(end - start)

        #     st.write("**Answer:**", response["answer"])
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input (like ChatGPT)
        user_input = st.chat_input("Ask a question")
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get assistant response
            start = time.time()
            response = chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "user_session"}}
            )
            end = time.time()

            answer = response["answer"]

            # Track analytics
            st.session_state.analytics["questions"] += 1
            st.session_state.analytics["response_times"].append(end - start)

            # Add assistant message to history
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

def admin_page():
    st.title("⚙️ Admin Page")
    st.write("Configure sources, prompts, and LLM settings.")

    # Choose LLM
    model_choice = st.selectbox(
        "Select LLM Model",
        ["Gemma2-9b-It", "deepseek-r1-distill-llama-70b"],
        index=["Gemma2-9b-It", "deepseek-r1-distill-llama-70b"].index(st.session_state.config["llm_model"])
    )
    st.session_state.config["llm_model"] = model_choice

    # Edit system prompt
    prompt_text = st.text_area("System Prompt", st.session_state.config["system_prompt"], height=200)
    st.session_state.config["system_prompt"] = prompt_text

    st.success("✅ Admin settings saved.")


def analytics_page():
    st.title("📊 Analytics Page")
    st.write("Usage and evaluation metrics")

    total_questions = st.session_state.analytics["questions"]
    response_times = st.session_state.analytics["response_times"]

    st.metric("Total Questions Asked", total_questions)
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        st.metric("Average Response Time (s)", round(avg_time, 2))
        st.line_chart(response_times)
    else:
        st.info("No queries asked yet.")


# -------------------------------
# Sidebar Navigation
# -------------------------------
page = st.sidebar.radio("Navigate", ["User Page", "Admin Page", "Analytics Page"])

if page == "User Page":
    user_page()
elif page == "Admin Page":
    admin_page()
else:
    analytics_page()
