# Databricks notebook source
# MAGIC %md
# MAGIC **NOTE:** You will need to first set up a [Vector Search endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) and [Vector Search index](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index) in order to run this chain. Please see [1_load_pdf_to_vector_index]($1_load_pdf_to_vector_index) to set up this infrastucture.

# COMMAND ----------

# DBTITLE 1,Databricks Rag Studio Installer
# MAGIC %pip install databricks-rag-studio databricks-vectorsearch mlflow>=2.13 langchain==0.2.0 langchain_core==0.2.0 langchain_community==0.2.0

# COMMAND ----------

# Before logging this chain using the driver notebook, you need to comment out this line.
dbutils.library.restartPython() 

# COMMAND ----------

# DBTITLE 1,Import packages
from databricks import rag
from operator import itemgetter
import mlflow
import os

from databricks.vector_search.client import VectorSearchClient

from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable MLflow Tracing
# MAGIC
# MAGIC Enabling MLflow Tracing is required to:
# MAGIC - View the chain's trace visualization in this notebook
# MAGIC - Capture the chain's trace in production via Inference Tables
# MAGIC - Evaluate the chain via the Mosaic AI Evaluation Suite

# COMMAND ----------

mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chain helper functions

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

# COMMAND ----------


############
# Get the configuration YAML
############
model_config = mlflow.models.ModelConfig(development_config="2_rag_chain_config.yaml")

############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index"),
)
vector_search_schema = model_config.get("vector_search_schema")

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
).as_retriever(search_kwargs=model_config.get("vector_search_parameters"))

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
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
    chunk_template = model_config.get("chunk_template")
    chunk_contents = [chunk_template.format(chunk_text=d.page_content) for d in docs]
    return "".join(chunk_contents)


############
# Prompt Template for generation
############
prompt = PromptTemplate(
    template=model_config.get("chat_prompt_template"),
    input_variables=model_config.get("chat_prompt_template_variables"),
)

############
# FM for generation
############
model = ChatDatabricks(
    endpoint=model_config.get("chat_endpoint"),
    extra_params=model_config.get("chat_endpoint_parameters"),
)

############
# RAG Chain
############
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)


# COMMAND ----------

############
# Test the Chain
############
model_input_sample = {
    "messages": [
        {
            "role": "user",
            "content": "What is DSPY?",
        }
    ]
}

# Uncomment this line to test the chain locally
# Private Preview workaround: We suggest commenting this line out before you log the model.  This is not strictly necessary but doing so will prevent additional MLflow traces from being show when calling mlflow.langchain.log_model(...).
# chain.invoke(model_input_sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tell MLflow logging where to find your chain.
# MAGIC
# MAGIC `mlflow.models.set_model(model=...)` function specifies the LangChain chain to use for evaluation and deployment.  This is required to log this chain to MLflow with `mlflow.langchain.log_model(...)`.

# COMMAND ----------

mlflow.models.set_model(model=chain)
