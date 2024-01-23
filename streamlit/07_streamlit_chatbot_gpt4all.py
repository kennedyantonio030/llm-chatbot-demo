import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


@st.cache_resource
def create_llm_instance():
    # Set up the prompt template
    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Set up the local instace of the LLM
    local_path = "/Users/brianroepke/Library/Application Support/nomic.ai/GPT4All/mistral-7b-openorca.Q4_0.gguf"  # noqa: E501
    callbacks = [StreamingStdOutCallbackHandler()]
    memory = ConversationBufferMemory()
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, streaming=True)  # noqa: E501
    # llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    llm_chain = ConversationChain(llm=llm)

    return llm_chain


llm_chain = create_llm_instance()

st.title("ChatGPT-Like Clone: GPT4All")


if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = "mistral-7b-openorca.Q4_0.gguf"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = llm_chain.invoke(prompt)
        full_response += response["response"]
        message_placeholder.markdown(full_response + "▌")
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )  # noqa: E501


# st.write(llm_chain.memory.buffer)
