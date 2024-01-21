import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


# Page title
st.set_page_config(page_title='🦜🔗 Ask the Data App')
st.title('🦜🔗 Ask the Data App')

# Get the OpenAI Key from Env Vars
openai_api_key = os.environ.get("OPEN_AI_KEY")

# Get the snowflake connection
conn = st.connection("snowflake")


@st.cache_data(ttl=3600)
def get_snowflake_table(_conn, table_name):
    snowflake_table = _conn.session()
    return snowflake_table.table(table_name).to_pandas()


# Fetch the table
df = get_snowflake_table(conn, "picks_twenty_four")


# Generate LLM response
def generate_response(df_input, input_query):
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.2,
        openai_api_key=openai_api_key
    )

    df = df_input
    # Create Pandas DataFrame Agent
    agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    # Perform Query using the Agent
    response = agent.run(input_query)
    return st.success(response)


# Input widgets
question_list = [
    "How many rows are there?",
    "What is the average age of the people in the DataFrame?",
    "Who is the oldest person?",
    "Other",
]
query_text = st.selectbox(
    "Select an example query:", question_list
)


# App logic
if query_text == "Other":
    query_text = st.text_input(
        "Enter your query:",
        placeholder="Enter query here ...",
    )
if not openai_api_key.startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
if openai_api_key.startswith("sk-") and (len(df) > 0):
    st.header("Output")
    generate_response(df, query_text)

    st.dataframe(df)
