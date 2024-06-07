# Databricks notebook source
# MAGIC %md
# MAGIC **This notebook shows you how to use Databricks' Evaluation Suite *without* RAG studio e.g., for a chain that you have deployed already.**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Evaluation Harness supports 2 approaches to provide your chain’s outputs in order to generate quality/cost/latency Metrics.   This notebook shows you approach #2.
# MAGIC
# MAGIC | Approach | Description | When to use |
# MAGIC | --- | --- | --- |
# MAGIC | 1 | **Eval Harness runs a chain on your behalf.** Pass a reference to the chain itself so Evaluation Harness can generate the outputs on your behalf | - Your chain is logged using MLflow w/ MLflow Tracing enabled <br> - Your chain is available as a Python function in your local notebook |
# MAGIC | 2 | **Run chain yourself, pass outputs to Eval Harness.** Run the chain being evaluated yourself, capturing the chain’s outputs and passing the outputs as a Pandas DataFrame | - Your chain is developed outside of Databricks <br> - You want to evaluate outputs from a chain already running in production <br> - You are testing different evaluation / LLM Judge configurations and your chain doesn’t produce deterministic outputs (e.g., LLM has a high temperature) |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Install pacakges

# COMMAND ----------

# DBTITLE 1,Install Evaluation Suite
# MAGIC %pip install databricks-agents 'mlflow>=2.13'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Evaluation Suite
import mlflow
import pandas as pd

# COMMAND ----------

# MAGIC %md # Step 1: Gather your chain's outputs (and optionally ground truth) to evaluate
# MAGIC
# MAGIC This first example shows you how to run evaluation on a single question from your chain's logs.

# COMMAND ----------

# MAGIC %md
# MAGIC First, you need to ETL your chain's logs into the Evaluation Harness input schema. `evaluate(...)` takes a parameter `data` that is a Pandas DataFrame containing your chain's outputs and, optionally, your Evaluation Set w/ ground truth.  Let's look at a few examples.
# MAGIC
# MAGIC
# MAGIC *- For full details on the schema, view the `Evaluation Harness Input Schema` section of the RAG Studio documentation.*<br/>
# MAGIC *- For full details on the metrics available, view the `LLM Judges & Metrics` section of the RAG Studio documentation.*
# MAGIC
# MAGIC These Dictionary-based examples are provided just to show the schema. You do NOT have to start from a Dictionary - you can use any existing Pandas or Spark DataFrame with this schema.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Level 1: Chain's request & response 
# MAGIC
# MAGIC **Required data:**
# MAGIC - Chain logs
# MAGIC   - `request`
# MAGIC   - `response`
# MAGIC
# MAGIC **Metrics available:**
# MAGIC - `response/llm_judged/relevance_to_query_rating/average`
# MAGIC - `response/llm_judged/harmfulness_rating/average`
# MAGIC - `chain/request_token_count`
# MAGIC - `chain/response_token_count`
# MAGIC - Customer-defined LLM judges

# COMMAND ----------

level_1_data = [
    {
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "response": "reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.",
    }]

#### Convert Dictionary to a Pandas DataFrame
level_1_data_df = pd.DataFrame(level_1_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Level 2: Complete chain logs w/ retrieval info
# MAGIC
# MAGIC **Required data:**
# MAGIC - Chain logs
# MAGIC   - `request`
# MAGIC   - `response`
# MAGIC   - `retrieved_context`
# MAGIC
# MAGIC **Metrics available:**
# MAGIC - Level 1 + 
# MAGIC   - `retrieval/llm_judged/chunk_relevance_precision/average`
# MAGIC   - `response/llm_judged/groundedness_rating/average`
# MAGIC

# COMMAND ----------

####
# Retrieval context is available from your chain logs
####
level_2_data = [
    {
        "request_id": "your-request-id", # optional, but useful for tracking
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "response": "reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.",
        "retrieved_context": [
            {
                # In `retrieved_context`, `content` is optional, but DOES deliver additional functionality if provided (the Databricks Context Relevance LLM Judge will run to check the `content`'s relevance to the `request`).
                "content": "reduceByKey reduces the amount of data shuffled by merging values before shuffling.",
                "doc_uri": "doc_uri_2_1",
            },
            {
                "content": "groupByKey may lead to inefficient data shuffling due to sending all values across the network.",
                "doc_uri": "doc_uri_6_extra",
            },
        ],
    }]

#### Convert Dictionary to a Pandas DataFrame
level_2_data_df = pd.DataFrame(level_2_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Level 3: Complete chain logs w / labeled ground truth answer
# MAGIC
# MAGIC **Required data:**
# MAGIC - Chain logs
# MAGIC   - `request`
# MAGIC   - `response`
# MAGIC   - `retrieved_context`
# MAGIC - Evaluation set
# MAGIC   - `expected_response`
# MAGIC
# MAGIC **Metrics available:**
# MAGIC - Level 2 + 
# MAGIC   - `response/llm_judged/correctness_rating/average`

# COMMAND ----------


####
# Ground truth is available
####
level_3_data  = [
    {
        "request_id": "your-request-id", 
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "expected_response": "There's no significant difference.",
        "response": "reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.",
        "retrieved_context": [
            {
                # In `retrieved_context`, `content` is optional, but DOES deliver additional functionality if provided (the Databricks Context Relevance LLM Judge will run to check the `content`'s relevance to the `request`).
                "content": "reduceByKey reduces the amount of data shuffled by merging values before shuffling.",
                "doc_uri": "doc_uri_2_1",
            },
            {
                "content": "groupByKey may lead to inefficient data shuffling due to sending all values across the network.",
                "doc_uri": "doc_uri_6_extra",
            },
        ],
    }]

#### Convert Dictionary to a Pandas DataFrame
level_3_data_df = pd.DataFrame(level_3_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Level 4: Complete chain logs w/ labeled ground truth answer & retrieval context
# MAGIC
# MAGIC **Required data:**
# MAGIC - Chain logs
# MAGIC   - `request`
# MAGIC   - `response`
# MAGIC   - `retrieved_context`
# MAGIC - Evaluation set
# MAGIC   - `expected_response`
# MAGIC   - `expected_retrieved_context`
# MAGIC
# MAGIC **Metrics available:**
# MAGIC - Level 3 + 
# MAGIC   - `retrieval/ground_truth/document_recall/average`
# MAGIC   - `retrieval/ground_truth/document_precision/average`

# COMMAND ----------

####
# Ground truth is available
####
level_4_data  = [
    {
        "request_id": "your-request-id", 
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "expected_retrieved_context": [
            {
                "doc_uri": "doc_uri_2_1",
            },
            {
                "doc_uri": "doc_uri_2_2",
            },
        ],
        "expected_response": "There's no significant difference.",
        "response": "reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.",
        "retrieved_context": [
            {
                # In `retrieved_context`, `content` is optional, but DOES deliver additional functionality if provided (the Databricks Context Relevance LLM Judge will run to check the `content`'s relevance to the `request`).
                "content": "reduceByKey reduces the amount of data shuffled by merging values before shuffling.",
                "doc_uri": "doc_uri_2_1",
            },
            {
                "content": "groupByKey may lead to inefficient data shuffling due to sending all values across the network.",
                "doc_uri": "doc_uri_6_extra",
            },
        ],
    }]

#### Convert Dictionary to a Pandas DataFrame

level_4_data_df = pd.DataFrame(level_4_data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Step 2: Run evaluation
# MAGIC
# MAGIC Once you have your data available, running evaluation is simple single-line command. 
# MAGIC
# MAGIC Here we use the `level_4_data_df` but you can replace this with any of the other example DataFrames above. 

# COMMAND ----------

# If you do not start a MLflow run, `evaluate(...) will start a Run on your behalf.
with mlflow.start_run(run_name="level_4_data"):
  evaluation_results = mlflow.evaluate(data=level_4_data_df, model_type="databricks-rag")

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3: Use the data & metrics
# MAGIC
# MAGIC Evaluation Harness produces several outputs:
# MAGIC - **Aggregated metric values across the entire Evaluation Set**
# MAGIC   - Average numerical result of each metric
# MAGIC - **Data about each question in the Evaluation Set**
# MAGIC   - In the same schema as the Evaluation Input
# MAGIC     - Inputs sent to the chain
# MAGIC     - All chain generated data used in evaluation e.g., response, retrieved_context, trace, etc
# MAGIC   - Numeric result of each metric e.g., 1 or 0, etc
# MAGIC   - Ratings & rationales from each Databricks and Customer-defined LLM judge
# MAGIC
# MAGIC These outputs are available in 2 locations:
# MAGIC 1. Stored inside the MLflow Run & Experiment as raw data & visualizations
# MAGIC 2. Returned as DataFrames & Dictionaries by `mlflow.evaluate(..)`
# MAGIC
# MAGIC *Note: The data is identical between these 2 locations, so which view you use is a matter of your preference.*

# COMMAND ----------

# MAGIC %md 
# MAGIC ### View inside MLflow
# MAGIC
# MAGIC Open the MLflow Experiment or Run using the links in the output cell above.  

# COMMAND ----------

# MAGIC %md
# MAGIC ### View as DataFrames & Dictionaries
# MAGIC #### Access aggregated metric values across the entire Evaluation Set

# COMMAND ----------

metrics_as_dict = evaluation_results.metrics

print("Aggregate metrics computed:")
display(metrics_as_dict)

# Sample usage
print(f"The average precision of the retrieval step is: {metrics_as_dict['retrieval/ground_truth/document_precision/average']}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Access the data produced on each question in the Evaluation Set

# COMMAND ----------

per_question_results_df = evaluation_results.tables['eval_results']
display(per_question_results_df)

print("Columns available for each question:")
print(per_question_results_df.columns)

# Show info about responses that are not grounded
per_question_results_df[per_question_results_df["response/llm_judged/groundedness_rating"] == True].display()
