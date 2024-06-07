# Databricks notebook source
# DBTITLE 1,Install RAG Studio packages
# MAGIC %pip install databricks-agents 'mlflow>=2.13' langchain==0.2.0 langchain_core==0.2.0 langchain_community==0.2.0

# COMMAND ----------

# Before logging this chain using the driver notebook, you need to comment out this line.
dbutils.library.restartPython() 

# COMMAND ----------

from operator import itemgetter
import mlflow
from langchain_community.chat_models import ChatDatabricks
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

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
# Your chain must accept an array of OpenAI-formatted messages as a `messages` parameter. 
# Schema: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chatmessage
# These helper functions help parse the `messages` array
############

# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hello World chain

# COMMAND ----------

# DBTITLE 1,Hello World Chain
############
# Prompt Template for generation
############
prompt = PromptTemplate(
    template="You are a hello world bot.  Respond with a reply to the user's question that is fun and interesting to the user.  User's question: {question}",
    input_variables=["question"],
)

############
# FM for generation
# ChatDatabricks accepts any /llm/v1/chat model serving endpoint
############
model = ChatDatabricks(
    endpoint="databricks-dbrx-instruct",
    extra_params={"temperature": 0.01, "max_tokens": 500},
)


############
# Simple chain
############
# RAG Studio requires the chain to return a string value.
chain = (
    {
        "question": itemgetter("messages")
        | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
    }
    | prompt
    | model
    | StrOutputParser()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the chain locally

# COMMAND ----------

# This is the same input your chain's REST API will accept.
question = {
    "messages": [
               {
            "role": "user",
            "content": "what is rag?",
        },
    ]
}

# Uncomment this line to test the chain locally
# Private Preview workaround: We suggest commenting this line out before you log the model.  This is not strictly necessary but doing so will prevent additional MLflow traces from being show when calling mlflow.langchain.log_model(...).
# chain.invoke(question)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tell MLflow logging where to find your chain.
# MAGIC
# MAGIC `mlflow.models.set_model(model=...)` function specifies the LangChain chain to use for evaluation and deployment.  This is required to log this chain to MLflow with `mlflow.langchain.log_model(...)`.

# COMMAND ----------

mlflow.models.set_model(model=chain)
