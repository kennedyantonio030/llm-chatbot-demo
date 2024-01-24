import streamlit as st
from langchain_openai import OpenAI
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)  # noqa: E501
from langchain.agents import create_react_agent, load_tools, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

st.title("Ollama Chat")
st.write("This is a simplified way of working with Streamlit message history.")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")


template = """Answer the following questions as best you can. 
You have access to the following tools: {tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# llm = ChatOllama(model="llama2:7b-chat", temperature=0.5)

llm = OpenAI(temperature=0.5)

# The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)


# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input("Ask a question here"):
    st.chat_message("human").write(prompt)
    config = {"configurable": {"session_id": "any"}}
    response = agent_executor.invoke(
        {"input": prompt, "chat_history": msgs}, config=config
    )
    st.chat_message("ai").write(response["output"])


# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
