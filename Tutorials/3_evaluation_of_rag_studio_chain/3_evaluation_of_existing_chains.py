# Databricks notebook source
# MAGIC %md
# MAGIC **This notebook shows you how to use Databricks' Evaluation Suite *with* RAG studio e.g., for a chain that you are building with RAG Studio.**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Evaluation Harness supports 2 approaches to provide your chain’s outputs in order to generate quality/cost/latency Metrics.   This notebook shows you approach #1.
# MAGIC
# MAGIC **For a fully worked examples that log and evaluate a chain, please see the [3_pdf_rag_with_single_turn_chat](../../RAG Cookbook/3_pdf_rag_with_single_turn_chat/) or [4_pdf_rag_with_multi_turn_chat](../../RAG Cookbook/4_pdf_rag_with_multi_turn_chat/) examples.**
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

# MAGIC %md # Step 1: Gather sample requests (and optionally ground truth) to evaluate

# COMMAND ----------

# MAGIC %md
# MAGIC First, you need to collect sample requests, optionally with ground truth, for evalaution to run on. `evaluate(...)` takes a parameter `data` that is a Pandas DataFrame your Evaluation Set, optionally, w/ ground truth.  Let's look at a few examples.
# MAGIC
# MAGIC *- For full details on the schema, view the `Evaluation Harness Input Schema` section of the RAG Studio documentation.*<br/>
# MAGIC *- For full details on the metrics available, view the `LLM Judges & Metrics` section of the RAG Studio documentation.*
# MAGIC
# MAGIC These Dictionary-based examples are provided just to show the schema. You do NOT have to start from a Dictionary - you can use any existing Pandas or Spark DataFrame with this schema.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Level A: Evaluation set contains just sample requests
# MAGIC
# MAGIC **Required data:**
# MAGIC - Evaluation set
# MAGIC   - `request`
# MAGIC
# MAGIC **Metrics available:**
# MAGIC - `response/llm_judged/relevance_to_query_rating`
# MAGIC - `response/llm_judged/harmfulness_rating/average`
# MAGIC - `retrieval/llm_judged/chunk_relevance_precision/average`
# MAGIC - `response/llm_judged/groundedness_rating/average`
# MAGIC - `chain/request_token_count`
# MAGIC - `chain/response_token_count`
# MAGIC - `chain/total_token_count`
# MAGIC - `chain/input_token_count`
# MAGIC - `chain/output_token_count`
# MAGIC - Customer-defined LLM judges
# MAGIC

# COMMAND ----------

level_A_data = [
    {
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
    }]

#### Convert Dictionary to a Pandas DataFrame
level_A_data_df = pd.DataFrame(level_A_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Level B: Evaluation set contains labeled ground truth answers
# MAGIC
# MAGIC **Required data:**
# MAGIC - Evaluation set
# MAGIC   - `request`
# MAGIC   - `expected_response`
# MAGIC
# MAGIC **Metrics available:**
# MAGIC - Level A metrics +
# MAGIC   - `response/llm_judged/correctness_rating/average`

# COMMAND ----------

level_B_data  = [
    {
        "request_id": "your-request-id", 
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "expected_response": "There's no significant difference.",
    }]

#### Convert Dictionary to a Pandas DataFrame
level_B_data_df = pd.DataFrame(level_B_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Level C: Evaluation set contains labeled ground truth answers & retrieval context
# MAGIC
# MAGIC **Required data:**
# MAGIC - Evaluation set
# MAGIC   - `request`
# MAGIC   - `expected_response`
# MAGIC   - `expected_retrieved_context`
# MAGIC
# MAGIC **Metrics available:**
# MAGIC - Level B metrics + 
# MAGIC   - `retrieval/ground_truth/document_recall/average`
# MAGIC   - `retrieval/ground_truth/document_precision/average`

# COMMAND ----------

level_C_data  = [
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
    }]

#### Convert Dictionary to a Pandas DataFrame

level_C_data_df = pd.DataFrame(level_C_data)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Step 2: Run evaluation
# MAGIC
# MAGIC Once you have your data available, running evaluation is simple single-line command. 
# MAGIC
# MAGIC Here we use the `level_C_data_df` but you can replace this with any of the other example DataFrames above. 
# MAGIC
# MAGIC There are 4 options for passing your chain to the Evaluation Harness.  
# MAGIC
# MAGIC - **Option 1.** Reference to a MLflow logged model in the current MLflow Experiment
# MAGIC     ```
# MAGIC     model = "runs:/6b69501828264f9s9a64eff825371711/chain" 
# MAGIC
# MAGIC     `6b69501828264f9s9a64eff825371711` is the run_id, `chain` is the artifact_path that was passed when calling mlflow.xxx.log_model(...).  
# MAGIC
# MAGIC     This value can be accessed via `model_info.model_uri` if you called model_info = mlflow.xxx.log_model(), where xxx is `langchain` or `pyfunc`.
# MAGIC     ```
# MAGIC
# MAGIC - **Option 2.** Reference to a Unity Catalog registered model 
# MAGIC     ```
# MAGIC     model = "models:/catalog.schema.model_name/1"  # 1 is the version number
# MAGIC     ```
# MAGIC
# MAGIC - **Option 3.** A PyFunc model that is loaded in the Notebook
# MAGIC     ```
# MAGIC     model = mlflow.pyfunc.load_model(...)
# MAGIC     ```
# MAGIC  
# MAGIC - **Option 4.** A local function in the Notebook
# MAGIC     ```
# MAGIC     def model_fn(model_input):
# MAGIC       # do stuff
# MAGIC       response = 'the answer!'
# MAGIC       return response
# MAGIC
# MAGIC     model = model_fn
# MAGIC     ```

# COMMAND ----------

# If you do not start a MLflow run, `evaluate(...) will start a Run on your behalf.
with mlflow.start_run(run_name="level_C_data"):
  evaluation_results = mlflow.evaluate(data=level_C_data_df, model="runs:/a828658a8c9f46eeb7ef346e65228394/chain", model_type="databricks-rag")

# COMMAND ----------

# MAGIC %md
# MAGIC **Option 1** is the most commonly used since you can simply call `evaluate(...)` on the output of `log_model(...)`:

# COMMAND ----------

# # Note this code will not directly run.


# with mlflow.start_run():
#     # Log the chain code + config + parameters to the run
#     logged_chain_info = mlflow.langchain.log_model(
#         lc_model=chain_notebook_path,
#         model_config=baseline_config,  # The configuration to test - this can also be a YAML file path rather than a Dict e.g., `chain_config_path`
#         artifact_path="chain",
#         input_example=input_example,
#         example_no_conversion=True,  # required to allow the schema to work
#         extra_pip_requirements=[  # temporary workaround needed during Private Preview
#             "databricks-rag-studio==0.2.0"
#         ],
#     )

#     # Evaluate the logged model
#     eval_results = mlflow.evaluate(
#         data=eval_set_df,
#         model=logged_chain_info.model_uri,
#         model_type="databricks-rag",
#     )

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
display(per_question_results_df.drop('trace', axis=1))

print("Columns available for each question:")
print(per_question_results_df.columns)

# Show info about responses that are not grounded
# per_question_results_df[per_question_results_df["response/llm_judged/groundedness_rating"] == True].display()
