import os
from dotenv import load_dotenv
import validators
import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_classic.callbacks import StreamlitCallbackHandler




############################################################################ Heading

st.title('RAG Chatbot: URL Interview')
st.write("Paste URL and Ask Chatbot Questions")

############################################################################ Sidebar

st.sidebar.title("Keys & Tokens")
groq_api_key=st.sidebar.text_input("Enter your Groq API Key: ",type="password")
hf_token=st.sidebar.text_input("Enter your HuggingFace Token: ",type="password")


############################################################################ App

if groq_api_key and hf_token:
    llm=ChatGroq(groq_api_key=groq_api_key,model='llama-3.1-8b-instant')

    # Get URL Input
    generic_url = st.text_input("URL", label_visibility="collapsed")

###########################################################################


    if not groq_api_key.strip() or not generic_url.strip():
        st.info("Please provide the URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (e.g. Youtube URL or Website URL)")
    else:
        try:
            with st.spinner("Processing..."):


                if generic_url:
                    st.session_state.session_id=st.text_input("Session ID", value="default_session")
                

                    def create_vector_embedding():

                        if 'store' not in st.session_state:
                            st.session_state.store={} 

                        
                        # 2. Loading the URL data
                        if "youtube.com" in generic_url or "youtu.be" in generic_url:
                            loader = YoutubeLoader.from_youtube_url(
                                generic_url, 
                                add_video_info=False, 
                            )
                        else:
                            loader = UnstructuredURLLoader(
                                urls=[generic_url],
                                ssl_verify=False,
                                headers={ 
                                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                                })
                            
                        docs=loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=5000, 
                            chunk_overlap=200, 
                            separators=["\n\n", "\n", ".", " ", ""]
                        )
                        chunks = text_splitter.split_documents(docs)
                        huggingface_embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')
                        st.session_state.vector_store = Chroma.from_documents(chunks,huggingface_embeddings)
                        st.session_state.retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})


                if "vector_store" not in st.session_state:
                    with st.spinner("Loading documents..."):
                        create_vector_embedding()

        except Exception as e:
            st.error(f"An error occurred: {e}")


############################################################################################

        contextualised_q_system_prompt=(
            """
                Given a chat history and the latest user question,
                which might reference the context in the chat history,
                formulate a standalone question which can be understood
                without the chat history.
                Do NOT answer the question.
                Just formulate it if needed.
                Otherwise, return it as it is.
            """
        )


        contextualised_q_prompt = ChatPromptTemplate.from_messages([
            ("system",contextualised_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}\n\n(Note: Please provide your answer in the same language I used for this question.)")
        ])


        history_aware_retriever=create_history_aware_retriever(llm,st.session_state.retriever,contextualised_q_prompt)


        # Q&A Prompt
        system_prompt=(
            """
                You are an assistant for consultation tasks.
                Use the following pieces of retrieved context to answer the questions.
                If you don't know, just say that you don't know.
                Keep the answer concise with the maximum of 3 sentences.

                {context}
            """
        )


        qa_prompt=ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])


        qa_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,qa_chain)


        def get_session_history(session:str)->BaseChatMessageHistory:
            if st.session_state.session_id not in st.session_state.store:
                st.session_state.store[st.session_state.session_id]=ChatMessageHistory()
            return st.session_state.store[st.session_state.session_id]
        

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )



        if "messages" not in st.session_state:
            st.session_state["messages"]=[{
                "role":"assistant",
                "content":"Hi, ask me anything about your content!"
            }]


        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        user_input = st.chat_input("Enter your question:")



        if user_input:
            session_history=get_session_history(st.session_state.session_id)
            st.session_state.messages.append({
                    "role":"user",
                    "content":user_input
                })
            st.chat_message("user").write(user_input)


            with st.chat_message("assistant"):
                with st.spinner("Analyzing documents and generating response..."):
                    streamlit_callback=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
                    response=conversational_rag_chain.invoke(
                        {"input":user_input},
                        config={
                            "configurable":{"session_id":st.session_state.session_id}
                        }
                    )

                st.write(response['answer'])
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":response['answer']
                })