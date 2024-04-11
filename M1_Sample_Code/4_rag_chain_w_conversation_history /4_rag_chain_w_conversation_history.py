# Databricks notebook source
# DBTITLE 1,Databricks Rag Studio Installer
# MAGIC %run ../wheel_installer

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch

# COMMAND ----------

# Before logging this chain using the driver notebook, you need to comment out this line.
dbutils.library.restartPython() 

# COMMAND ----------

# DBTITLE 1,Import packages
from operator import itemgetter

from databricks import rag
from databricks.vector_search.client import VectorSearchClient
from langchain.schema.runnable import RunnableLambda
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# COMMAND ----------

############
# Private Preview Feature MLflow Tracing
# RAG Studio dependes on MLflow to show a trace of your chain. The trace can help you easily debug your chain and keep track of inputs & responses your chain performs well or performs poorly.
############

import mlflow
mlflow.langchain.autolog()

# COMMAND ----------

############
# RAG Studio requires your chain to accept an array of OpenAI-formatted messages as a `messages` parameter. Schema: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chatmessage
# These helper functions help parse the `messages` array
############
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


############
# Get the configuration YAML
############
rag_config = rag.RagConfig("4_rag_chain_w_conversation_history_config.yaml")


############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=rag_config.get("vector_search_endpoint_name"),
    index_name=rag_config.get("vector_search_index"),
)
vector_search_schema = rag_config.get("vector_search_schema")

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column=vector_search_schema.get("chunk_text"),
    columns=[
        vector_search_schema.get("primary_key"),
        vector_search_schema.get("chunk_text"),
        vector_search_schema.get("document_source"),
    ],
).as_retriever(search_kwargs=rag_config.get("vector_search_parameters"))

############
# Enable the RAG Studio Review App to properly display retrieved chunks
############
rag.set_vector_search_schema(
    primary_key=vector_search_schema.get("primary_key"),
    text_column=vector_search_schema.get("chunk_text"),
    doc_uri=vector_search_schema.get(
        "document_source"
    ),  # Review App uses `doc_uri` to display chunks from the same document in a single view
)

############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = rag_config.get("chunk_template")
    chunk_contents = [chunk_template.format(chunk_text=d.page_content) for d in docs]
    return "".join(chunk_contents)


############
# Prompt Template for generation
############
prompt = PromptTemplate(
    template=rag_config.get("chat_prompt_template"),
    input_variables=rag_config.get("chat_prompt_template_variables"),
)

############
# Prompt Template for query rewriting to allow converastion history to work
############
query_rewrite_prompt = PromptTemplate(
    template=rag_config.get("query_rewriter_prompt_template"),
    input_variables=rag_config.get("query_rewriter_prompt_template_variables"),
)


############
# FM for generation
############
model = ChatDatabricks(
    endpoint=rag_config.get("chat_endpoint"),
    extra_params=rag_config.get("chat_endpoint_parameters"),
)

############
# RAG Chain
############
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        # "context": itemgetter("messages")
        # | RunnableLambda(extract_user_query_string)
        # | vector_search_as_retriever
        # | RunnableLambda(format_context),
    }
      | RunnablePassthrough()
    | {
            "context": query_rewrite_prompt
            | model
            | StrOutputParser()
            | vector_search_as_retriever
            | RunnableLambda(format_context),
            # "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question"),
        }
    | prompt
    | model
    | StrOutputParser()
)

############
# Test the Chain
############
model_input_sample = {
    "messages": [
        {
            "role": "user",
            "content": "What is ARES?",
        },
        {
            "role": "assistant",
            "content": "a thing for rag",
        },
        {
            "role": "user",
            "content": "how do you use it",
        }
    ]
}
print(chain.invoke(model_input_sample))

############
# Tell RAG Studio about the chain - required for logging, but not local testing of this chain
############
rag.set_chain(chain)