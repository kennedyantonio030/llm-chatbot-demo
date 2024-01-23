import streamlit as st
from langchain_community.llms import GPT4All
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate


@st.cache_resource
def create_llm_instance():
    # Set up the prompt template
    template = """
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant:
    """  # noqa: E501

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)  # noqa: E501

    # Set up the local instace of the LLM
    local_path = "/Users/brianroepke/Library/Application Support/nomic.ai/GPT4All/mistral-7b-openorca.Q4_0.gguf"  # noqa: E501
    llm = GPT4All(model=local_path, verbose=True)

    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory(human_prefix="Human"),
            )

    return conversation


conversation = create_llm_instance()

st.title("GPT4All Chat")
st.write("This method uses traditional message history in order to retain context.")  # noqa: E501


if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = "mistral-7b-openorca.Q4_0.gguf"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for token in conversation.stream(user_input):
            full_response += token["response"]
            message_placeholder.markdown(full_response + "▌")
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )  # noqa: E501
