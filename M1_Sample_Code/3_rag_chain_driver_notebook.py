# Databricks notebook source
# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %run ./wheel_installer
# MAGIC %pip install --quiet mlflow[databricks] mlflow[skinny]

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

import os
import mlflow
from databricks import rag_studio, rag_eval, rag
import json
import html

### START: Ignore this code, temporary workarounds given the Private Preview state of the product
from mlflow.utils import databricks_utils as du
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "false"

def parse_deployment_info(deployment_info):
  browser_url = du.get_browser_hostname()
  message = f"""Deployment of {deployment_info.model_name} version {deployment_info.model_version} initiated.  This can take up to 15 minutes and the Review App & REST API will not work until this deployment finishes. 

  View status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}
  Review App: {deployment_info.rag_app_url}"""
  return message
### END: Ignore this code, temporary workarounds given the Private Preview state of the product

# COMMAND ----------

# MAGIC %run ./RAG_Experimental_Code

# COMMAND ----------

# MAGIC %md
# MAGIC # Configure the driver notebook 

# COMMAND ----------

# DBTITLE 1,Setup
############
# Specify the full path to the chain notebook & config YAML
############

# Assuming your chain notebook is in the current directory, this helper line grabs the current path, prepending /Workspace/
# Limitation: RAG Studio does not support logging chains stored in Repos
current_path = '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

chain_notebook_file = "3_rag_chain"
chain_config_file = "3_rag_chain_config.yaml"
chain_notebook_path = f"{current_path}/{chain_notebook_file}"
chain_config_path = f"{current_path}/{chain_config_file}"

print(f"Saving chain from: {chain_notebook_path}, config from: {chain_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Log the chain

# COMMAND ----------

# DBTITLE 1,Log the model

############
# Log the chain to the Notebook's MLflow Experiment inside a Run
# The model is logged to the Notebook's MLflow Experiment as a run
############

logged_chain_info = rag_studio.log_model(code_path=chain_notebook_path, config_path=chain_config_path)

# Optionally, tag the run to save any additional metadata
with mlflow.start_run(run_id=logged_chain_info.run_id):
  mlflow.set_tag(key="your_custom_tag", value="info_about_chain")

# Save YAML config params to the Run for easy filtering / comparison later(requires experimental import)
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
RagConfig(chain_config_path).experimental_log_to_mlflow_run(run_id=logged_chain_info.run_id)

print(f"MLflow Run: {logged_chain_info.run_id}")
print(f"Model URI: {logged_chain_info.model_uri}")

############
# If you see this error, go to your chain code and comment out all usage of `dbutils`
############
# ValueError: The file specified by 'code_path' uses 'dbutils' command which are not supported in a chain model. To ensure your code functions correctly, remove or comment out usage of 'dbutils' command.

# COMMAND ----------

# MAGIC %md
# MAGIC # Test the model locally & view the trace

# COMMAND ----------

# DBTITLE 1,Local Model Testing and Tracing
############
# Test the model locally
# This is the same input that the REST API will accept once deployed.
############

model_input = {
    "messages": [
        {
            "role": "user",
            "content": "Hello world!!",
        },
        
    ]
}

loaded_model = mlflow.langchain.load_model(logged_chain_info.model_uri)

# Run the model to see the output
# loaded_model.invoke(question)


############
# Experimental: View the trace
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############
json_trace = experimental_get_json_trace(loaded_model, model_input)

json_string = json.dumps(json_trace, indent=4)

# Escape HTML characters to avoid XSS or rendering issues
escaped_json_string = html.escape(json_string)

# Generate HTML with the escaped JSON inside <pre> and <code> tags
pretty_json_html = f"<html><body><pre><code>{escaped_json_string}</code></pre></body></html>"

# To use the HTML string in a context that renders HTML, 
# such as a web application or a notebook cell that supports HTML output
displayHTML(pretty_json_html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the chain

# COMMAND ----------

# MAGIC %md
# MAGIC ## First, build an evaluation set
# MAGIC
# MAGIC The evaluation set represents the human-annotated ground truth data.
# MAGIC
# MAGIC | Column Name                  | Type                                              | Required? | Comment                                                                                                                                                  |
# MAGIC |------------------------------|---------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC | request_id                   | STRING                                            | Either `request_id` or `request` is required        | Id of the request (question)                                                                                                                             |
# MAGIC | request                     | STRING                                            |   Either `request_id` or `request` is required        | A request (question) to the RAG app, e.g., “What is Spark?”                                                                                              |
# MAGIC | expected_response            | STRING                                            |           | (Optional) The expected answer to this question                                                                                                          |
# MAGIC | expected_retrieval_context   | ARRAY<STRUCT<doc_uri: STRING, content: STRING>>   |           | (Optional) The expected retrieval context. The entries are ordered in descending rank. Each entry can record the URI of the retrieved doc and optionally the (sub)content that was retrieved. |
# MAGIC

# COMMAND ----------

############
# Expiermental: you can query the model to iteratively build your evaluation set
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############

eval_dataset = []
expected_retrieval_context = None
request_id = "sample_1"
request = ""
expected_response = ""

for step in json_trace["steps"]:
  if step['type'] == "RETRIEVAL":
    expected_retrieval_context = step['retrieval']['chunks']
    request = step['retrieval']['query_text']
  elif step['type'] == "LLM_GENERATION":
    expected_response = step['text_generation']['generated_text']

ground_truth_example = {
        "request_id": request_id,
        "request": request,
        # Retrieval context is optional
        "expected_retrieval_context": expected_retrieval_context,
        # Expected response is optional
        "expected_response": expected_response,
    }

ground_truth_example

# COMMAND ----------

# DBTITLE 1,Sample Evaluation Dataset Loader
############
# If you have a known set of queries, you can build the evaluation dataset manually
# Alternatively, you can create the evaluation dataset using Spark/SQL - it is simply an Delta Table with the above schema
############

eval_dataset = [
    {
        "request_id": "sample_request_1",
        "request": "What is ARES?",
        # Expected retrieval context is optional, if not provided, RAG Studio will use LLM judge to assess each retrieved context
        "expected_retrieval_context": [
            {
                        "chunk_id": "9517786ecadf3e0c75e3cd4ccefdced5",
                        "doc_uri": "dbfs:/Volumes/rag/ericp_m1/matei_pdf/2311.09476.pdf",
                        "content": "..."
                        
                    },
                    {
                        "chunk_id": "e8825fe982f7fd190ad828a307d7f280",
                        "doc_uri": "dbfs:/Volumes/rag/ericp_m1/matei_pdf/2311.09476.pdf",
                        "content": "..."
                        
                    },
                    {
                        "chunk_id": "e47b43c9c8f8ce11d78342c49ddbea07",
                        "doc_uri": "dbfs:/Volumes/rag/ericp_m1/matei_pdf/2311.09476.pdf",
                        "content": "..."
                        
                    }
        ],
        # Expected response is optional
        "expected_response": "ARES is an Automated RAG Evaluation System for evaluating retrieval-augmented generation (RAG) systems along the dimensions of context relevance, answer faithfulness, and answer relevance. It uses synthetic training data to finetune lightweight LM judges to assess the quality of individual RAG components and utilizes a small set of human-annotated datapoints for prediction-powered inference (PPI) to mitigate potential prediction errors.",
    }
]

############
# Turn the eval dataset into a Delta Table
############
uc_catalog = "catalog"
uc_schema = "schema"
eval_table_name = "sample_eval_set"
eval_table_fqdn = f"{uc_catalog}.{uc_schema}.{eval_table_name}"

df = spark.read.json(spark.sparkContext.parallelize(eval_dataset))
df.write.format("delta").option("mergeSchema", "true").mode("overwrite").saveAsTable(
    eval_table_fqdn
)
print(f"Loaded eval set to: {eval_table_fqdn}")

# COMMAND ----------

# DBTITLE 1,YAML Assessment Config Parser
import yaml

############
# Default evaluation configuration
############
config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval",
            "assessments": [
                "harmful",
                "faithful_to_context",
                "relevant_to_question_and_context",
                "relevant_to_question",
                "answer_good",
                "context_relevant_to_question",
            ],
        }
    ]
}


############
# Currently, evaluation is slow with the Databricks provided LLM judge due to a limitation we are working to remove.  You can temporarily use any Model Serving endpoint to overcome this limitation, including DBRX.
############
config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval_dbrx",
            "endpoint_name": "endpoints:/databricks-dbrx-instruct",
            "assessments": [
                "harmful",
                "faithful_to_context",
                "relevant_to_question_and_context",
                "relevant_to_question",
                "answer_good",
                "context_relevant_to_question",
            ],
        }
    ]
}

config_yml = yaml.dump(config_json)
config_yml

# COMMAND ----------

# DBTITLE 1,Machine Learning Experiment Tracker
############
# Run evaluation, logging the results to a sub-run of the chain's MLflow run
############
with mlflow.start_run(logged_chain_info.run_id):
  evaluation_results = rag_eval.evaluate(eval_set_table_name=eval_table_fqdn, model_uri=logged_chain_info.model_uri, config=config_yml)

  ############
  # Experimental: Log evaluation results to MLflow.  Note you can also use the dashboard produced by RAG Studio to view metrics/debug quality - it has more advanced functionality.
  # Known issues: Can only be run once per run_id.
  # ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
  ############
  experimental_add_metrics_to_run(evaluation_results, evaluation_results.mlflow_run_id)
  experimental_add_eval_outputs_to_run(evaluation_results, evaluation_results.mlflow_run_id)
  experimental_add_eval_config_tags_to_run(evaluation_results, config_yml, evaluation_results.mlflow_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy the model to the Review App

# COMMAND ----------

# DBTITLE 1,Deploy the model
############
# To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.
############
uc_catalog = "catalog"
uc_schema = "schema"
model_name = "pdf_bot"
uc_model_fqdn = f"{uc_catalog}.{uc_schema}.{model_name}" 

mlflow.set_registry_uri('databricks-uc')
uc_registered_chain_info = mlflow.register_model(logged_chain_info.model_uri, uc_model_fqdn)

# COMMAND ----------

############
# Deploy the chain to:
# 1) Review App so you & your stakeholders can chat with the chain & given feedback via a web UI.
# 2) Chain REST API endpoint to call the chain from your front end
# 3) Feedback REST API endpoint to pass feedback back from your front end.
############

deployment_info = rag_studio.deploy_model(uc_model_fqdn, uc_registered_chain_info.version)
print(parse_deployment_info(deployment_info))

# Note: It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

# DBTITLE 1,View deployments
############
# If you lost the deployment information captured above, you can find it using list_deployments()
############
deployments = rag_studio.list_deployments()
for deployment in deployments:
  if deployment.model_name == uc_model_fqdn and deployment.model_version==uc_registered_chain_info.version:
    print(parse_deployment_info(deployment))
