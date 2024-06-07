# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 3. PDF RAG w/ multi-turn chat Driver Notebook
# MAGIC
# MAGIC This notebook demonstrates how to use Databricks RAG Studio to log and evaluate a RAG chain with a [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) retrieval component. Note that you will have to first create a vector search endpoint, and a vector search index in order to run this notebook. Please first run the [`1_load_pdf_to_vector_index` notebook]($1_load_pdf_to_vector_index) first to set up this infrastructure. Refer to [the following documentation](https://docs.databricks.com/en/generative-ai/vector-search.html#how-to-set-up-vector-search) for more information on this. 
# MAGIC
# MAGIC This notebook covers the following steps:
# MAGIC
# MAGIC 1. Install required libraries and import required modules
# MAGIC 3. Define paths for the chain notebook and config YAML
# MAGIC 4. Log the chain to MLflow and test it locally, viewing the trace
# MAGIC 5. Evaluate the chain using an eval dataset
# MAGIC 6. Deploy the chain

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies

# COMMAND ----------

# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %pip install databricks-agents 'mlflow>=2.13'

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %run ../../utilities/prpr_shared_funcs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import os
import mlflow
from databricks import agents
import pandas as pd
# Use Unity Catalog as the model registry
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the Unity Catalog location where the chain will be logged

# COMMAND ----------

# Create widgets for user input
dbutils.widgets.text("uc_catalog", "", "Unity Catalog")
dbutils.widgets.text("uc_schema", "", "Unity Catalog Schema")
dbutils.widgets.text("model_name", "pdf_rag_bot_multi_turn", "Model Name")

# Retrieve the values from the widgets
uc_catalog = dbutils.widgets.get("uc_catalog")
uc_schema = dbutils.widgets.get("uc_schema")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define your evaluation set
# MAGIC
# MAGIC In order to deliver high-quality RAG applications, Databricks recommends following an evaluation-driven approach to development.  To start your development process, we suggest starting with ~5 to 10 examples of questions that your users will expect your RAG application to answer correctly.  Over the course of your development process, you will expand this evaluation set.
# MAGIC
# MAGIC | Column Name                  | Type                                              | Required? | Comment                                                                                                                                                  |
# MAGIC |------------------------------|---------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC | request_id                   | STRING                                            | Either `request_id` or `request` is required        | Id of the request (question)                                                                                                                             |
# MAGIC | request                     | STRING                                            |   Either `request_id` or `request` is required        | A request (question) to the RAG app, e.g., “What is Spark?”                                                                                              |
# MAGIC | expected_response            | STRING                                            |           | (Optional) The expected answer to this question                                                                                                          |
# MAGIC | expected_retrieved_context   | ARRAY<STRUCT<doc_uri: STRING, content: STRING>>   |           | (Optional) The expected retrieval context. The entries are ordered in descending rank. Each entry can record the URI of the retrieved doc and optionally the (sub)content that was retrieved. |
# MAGIC

# COMMAND ----------

eval_set = [
    {
        "request_id": "synthetic_97496aa16cefcde44bc4ad97f00b9f85",
        "request": "Did GPT-4's opinion response rate increase or decrease by June 2023?",
        "expected_response": "Decrease", # Optional
        "expected_retrieved_context": [ # Optional
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2307.09009.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_446c080e9f62a0e9c919a53c810c5814",
        "request": "What metric quantifies LLM generated answer variance?",
        "expected_response": "Mismatch quantifies how often LLM generated answers vary for the same prompt.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2307.09009.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_2969374e1bdf95f80879f3bc425028f8",
        "request": "What defines a happy number?",
        "expected_response": "A happy number reaches 1 when replaced by the sum of the squares of its digits repeatedly.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2307.09009.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_e8269ec16e376a5dea296ebaf0307912",
        "request": "What dataset does the multi-hop QA study use?",
        "expected_response": "HotPotQA",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2310.03714.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_51ba79a962b751ea24039ce401c911e1",
        "request": "What do DSPy signatures abstract?",
        "expected_response": "Prompting and finetuning processes.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2310.03714.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_b839bcfd0bd5d8cef06b11f1013f2341",
        "request": "What are DSPy modules parameterized to learn?",
        "expected_response": "Prompting, finetuning, augmentation, and reasoning techniques.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2310.03714.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_0f3b364c0957291ccb93e321f67fdf16",
        "request": "What does DSPy's BootstrapFewShot simulate?",
        "expected_response": "A teacher program or the zero-shot version of the program being compiled.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2310.03714.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_d163c20a260349712a8574cbf58f954e",
        "request": "What is Torch?",
        "expected_response": "A modular machine learning software library.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2310.03714.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_9565f255bfd9559df84a199ee4624dfd",
        "request": "What mitigates accuracy loss in ARES's cross-domain applications?",
        "expected_response": "PPI mitigates the loss in accuracy.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2311.09476.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_c3818718cf09cfffbbcd2bfe21077436",
        "request": "What are DSPy's LM Assertions designed for?",
        "expected_response": "Expressing computational constraints on LMs in larger programs.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2312.13382.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_67de36b217cb3b215ce4ba706fa8b18f",
        "request": "What is Infer–Retrieve–Rank's current limitation?",
        "expected_response": "It requires one GPT-4 call per input document.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2401.12178.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_63ab00ec9cf9c64c507690534ef2084d",
        "request": "What does the DEMONSTRATE stage do in DSP?",
        "expected_response": "It prepares a list of demonstrations by selecting a subset of the training examples and bootstrapping new fields in them.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2212.14024.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_6a64839deb74baf7d5a9ffc713bd7f1d",
        "request": "What year was Stochastic Gradient Boosting published?",
        "expected_response": "2002",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2305.05176.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_f01c09870b6228654b46cece85498fdb",
        "request": "What does FrugalGPT aim to optimize?",
        "expected_response": "FrugalGPT aims to optimize task performance with LLM APIs under budget constraints.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/2305.05176.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_e1d0be25e73bd5602ed086e111e12545",
        "request": "What is a key trend in AI for 2024?",
        "expected_response": "Compound AI systems are a key trend.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/The Shift from Models to Compound AI Systems – The Berkeley Artificial Intelligence Research Blog.pdf",
                # "content": "...",
            }
        ],
    },
    {
        "request_id": "synthetic_7b4e1ada2a0d23174c433f3827ec7654",
        "request": "Why are compound AI systems preferred?",
        "expected_response": "They achieve state-of-the-art results more efficiently than monolithic models.",
        "expected_retrieved_context": [
            {
                "doc_uri": "dbfs:/Volumes/ep_05_08_release/rag/pdf_docs/The Shift from Models to Compound AI Systems – The Berkeley Artificial Intelligence Research Blog.pdf",
                # "content": "...",
            }
        ],
    },
]

# Turn the evaluation set into a Pandas Dataframe
eval_set_df = pd.DataFrame(eval_set)

display(eval_set_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Now, log the baseline RAG chain & evaluate it against this set of questions/answers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow logging input parameters

# COMMAND ----------

# DBTITLE 1,Setup
# Provide an example of the input schema that is used to set the MLflow model's signature
input_example = {
   "messages": [
       {
           "role": "user",
           "content": "What is Retrieval-augmented Generation?",
       }
   ]
}

# Specify the full path to the chain notebook
chain_notebook_file = "2_rag_chain_multi_turn"
chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)

# This is optional if you want to use the configuration stored in the YAML - this can be specified in a Dict as shown below
chain_config_file = "2_rag_chain_config.yaml"
chain_config_path = os.path.join(os.getcwd(), chain_config_file)

print(f"Chain notebook path: {chain_notebook_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Log & evaluate the chain
# MAGIC Log the chain to the Notebook's MLflow Experiment inside a Run. The model is logged to the Notebook's MLflow Experiment as a run.
# MAGIC
# MAGIC To learn more about the metrics that are computed, see the documentation.

# COMMAND ----------

# DBTITLE 1,Log the model
# Update this config from the output of the last cell of `1_load_pdf_vector_index`
# You can simply copy / paste the JSON that dumped from the last cell and replace the entire `baseline_config`.
baseline_config = {
        "vector_search_endpoint_name": "REPLACE ME WITH VALUE FROM 1_load_pdf_vector_index",
    "vector_search_index": "REPLACE ME WITH VALUE FROM 1_load_pdf_vector_index",
    "vector_search_schema": {
        "primary_key": "chunk_id",
        "chunk_text": "chunked_text",
        "document_source": "doc_uri",
    },
    "vector_search_parameters": {"k": 3},
    "chunk_template": "`{chunk_text}`\n",
    "chat_endpoint": "databricks-dbrx-instruct",
    "chat_prompt_template": "You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
    "chat_prompt_template_variables": ["context", "question"],
    "chat_endpoint_parameters": {"temperature": 0.01, "max_tokens": 500},
    "query_rewriter_prompt_template": "Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.\n\nChat history: {chat_history}\n\nQuestion: {question}",
    "query_rewriter_prompt_template_variables": ["chat_history", "question"],
}

with mlflow.start_run():
    # Log the chain code + config + parameters to the run
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,
        model_config=baseline_config,  # The configuration to test - this can also be a YAML file path rather than a Dict e.g., `chain_config_path`
        artifact_path="chain",
        input_example=input_example,
        example_no_conversion=True,  # required to allow the schema to work
    )

    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_set_df,
        model=logged_chain_info.model_uri,
        model_type="databricks-rag",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the results of evaluation
# MAGIC
# MAGIC To view the evaluation results, you can open the MLflow Experiment and view the data/metrics.
# MAGIC Alternatively, the metrics and evaluation results (including the LLM judge) below as dataframes.

# COMMAND ----------

# View the metrics computed over the entire evaluation set
print(eval_results.metrics)

# View the question-by-question results and LLM judged assessments.
display(eval_results.tables['eval_results'].drop(columns=["trace"]))

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy the model to the Review App to collect stakeholder feedback
# MAGIC
# MAGIC To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.

# COMMAND ----------

# DBTITLE 1,Register to UC
# Unity Catalog location
uc_model_fqn = f"{uc_catalog}.{uc_schema}.{model_name}"

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=uc_model_fqn )

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy the chain to:
# MAGIC 1. Review App so you & your stakeholders can chat with the chain & given feedback via a web UI.
# MAGIC 2. Chain REST API endpoint to call the chain from your front end.
# MAGIC 3. Feedback REST API endpoint to pass feedback back from your front end.
# MAGIC
# MAGIC **Note:** It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

deployment_info = agents.deploy(uc_model_fqn, uc_registered_model_info.version)

# Note: It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## View deployments
# MAGIC
# MAGIC If you have lost the deployment information captured above, you can find it using `list_deployments()`.

# COMMAND ----------

# DBTITLE 1,View deployments
deployments = agents.list_deployments()
for deployment in deployments:
  if deployment.model_name == uc_model_fqn and deployment.model_version==uc_registered_model_info.version:
    print(parse_deployment_info(deployment))
