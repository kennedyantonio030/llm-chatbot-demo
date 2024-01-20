import os
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.vectorstores import FAISS

OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")


conn = st.connection("snowflake")
snowflake_table = conn.session()
df = snowflake_table.table("nndb").to_pandas()
df = df.dropna(subset=["WIKI_DEATH_SUMMARY"]).head(100)

if len(df) is not None:
    loader = DataFrameLoader(df, "WIKI_DEATH_SUMMARY")
    data = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=0.0,
            model_name="gpt-3.5-turbo",
            openai_api_key=OPEN_AI_KEY,
        ),
        retriever=vectors.as_retriever(),
    )

    def conversational_chat(query):
        result = chain({"question": query,
                        "chat_history": st.session_state["history"]}
                       )
        st.session_state["history"].append((query, result["answer"]))

        return result["answer"]

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            "Hello ! Ask me anything about your DataFrame"
        ]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey ! 👋"]

    # container for the chat history
    response_container = st.container()
    # container for the user's text input
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Query:",
                placeholder="Talk about your csv data here (:",
                key="input"
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = conversational_chat(user_input)

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="adventurer",
                )
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="bottts"
                )
