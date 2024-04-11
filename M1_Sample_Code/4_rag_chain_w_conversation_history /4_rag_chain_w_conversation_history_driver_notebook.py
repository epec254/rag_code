# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # 4. RAG Chain with Conversation History Driver Notebook
# MAGIC 
# MAGIC This notebook demonstrates how to use Databricks RAG Studio to log and evaluate a RAG chain with conversation history. It assumes that you have already set up the necessary vector search endpoint and index. Please make sure to update the catalog and schema names in the notebook.
# MAGIC 
# MAGIC This notebook covers the following steps:
# MAGIC 
# MAGIC 1. Install required libraries and import required modules
# MAGIC 2. Define paths for the chain notebook and config YAML
# MAGIC 3. Log the chain to MLflow and test it locally, viewing the trace
# MAGIC 4. Evaluate the chain using an eval dataset
# MAGIC 5. Deploy the chain

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies

# COMMAND ----------

# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %run ../wheel_installer

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import json
import os

import mlflow
from databricks import rag, rag_eval, rag_studio

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

# MAGIC %run ../RAG_Experimental_Code

# COMMAND ----------

# MAGIC %md
# MAGIC # Define paths for chain notebook and config YAML

# COMMAND ----------

# DBTITLE 1,Setup
# Specify the full path to the chain notebook & config YAML
chain_notebook_file = "4_rag_chain_w_conversation_history"
chain_config_file = "4_rag_chain_w_conversation_history_config.yaml"

chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)
chain_config_path = os.path.join(os.getcwd(), chain_config_file)

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain config path: {chain_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Log the chain
# MAGIC Log the chain to the Notebook's MLflow Experiment inside a Run. The model is logged to the Notebook's MLflow Experiment as a run.

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
            "content": "What is ARES?",
        },
        {
            "role": "assistant",
            "content": "ARES is a thing for RAG.",
        },
        {
            "role": "user",
            "content": "How do you use it?",
        }
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