# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Hello World Driver Notebook
# MAGIC
# MAGIC This notebook demonstrates how to log, test, and deploy a simple "Hello World" chain using Databricks RAG Studio. It covers the following steps:
# MAGIC
# MAGIC 1. Install Databricks RAG Studio
# MAGIC 2. Import required modules
# MAGIC 3. Define path for the chain notebook
# MAGIC 4. Log the chain to MLflow and test it locally
# MAGIC 5. Register the chain as a model in Unity Catalog
# MAGIC 6. Deploy the chain
# MAGIC 7. View the deployed chain in the RAG Studio UI

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %run ../wheel_installer

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

# DBTITLE 1,Imports
import os

import mlflow
from databricks import rag_studio

mlflow.set_registry_uri('databricks-uc')

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

# MAGIC %md
# MAGIC ## Define paths for chain notebook

# COMMAND ----------

# DBTITLE 1,Setup
# Specify the full path to the chain notebook
chain_notebook_file = "1_hello_world_chain"
chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)

print(f"Chain notebook path: {chain_notebook_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the chain to MLflow and test locally
# MAGIC
# MAGIC Log the chain to the Notebook's MLflow Experiment inside a Run. The model is logged to the Notebook's MLflow Experiment as a run.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Log the model
logged_chain_info = rag_studio.log_model(code_path=chain_notebook_path)

print(f"MLflow Run: {logged_chain_info.run_id}")
print(f"Model URI: {logged_chain_info.model_uri}")

############
# If you see this error, go to your chain code and comment out all usage of `dbutils`
############
# ValueError: The file specified by 'code_path' uses 'dbutils' command which are not supported in a chain model. To ensure your code functions correctly, remove or comment out usage of 'dbutils' command.

# COMMAND ----------

# MAGIC %md
# MAGIC You can test the model locally. This is the same input that the REST API will accept once deployed.

# COMMAND ----------

# DBTITLE 1,Run the logged model locally
example_input = {
    "messages": [
        {
            "role": "user",
            "content": "Hello world!!",
        },
        {
            "role": "assistant",
            "content": "Hello back.",
        },
        {
            "role": "user",
            "content": "Hello again.",
        }
    ]
}

model = mlflow.langchain.load_model(logged_chain_info.model_uri)
model.invoke(example_input)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the chain

# COMMAND ----------

############
# Normally, you would now evaluate the chain, but lets skip ahead to deploying the chain so your stakeholders can use it via a chat UI.
############

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy chain
# MAGIC
# MAGIC To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.

# COMMAND ----------

# TODO: Change these values to your catalog and schema
uc_catalog = "catalog"
uc_schema = "schema"
model_name = "hello_world"
uc_model_fqdn = f"{uc_catalog}.{uc_schema}.{model_name}" 

uc_registered_chain_info = mlflow.register_model(logged_chain_info.model_uri, uc_model_fqdn)

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy the chain to:
# MAGIC 1. Review App so you & your stakeholders can chat with the chain & given feedback via a web UI.
# MAGIC 2. Chain REST API endpoint to call the chain from your front end.
# MAGIC 3. Feedback REST API endpoint to pass feedback back from your front end.
# MAGIC
# MAGIC **Note:** It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Deploy the model
deployment_info = rag_studio.deploy_model(uc_model_fqdn, uc_registered_chain_info.version)
print(parse_deployment_info(deployment_info))

# Note: It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## View deployments
# MAGIC
# MAGIC If you have lost the deployment information captured above, you can find it using `list_deployments()`.
# MAGIC

# COMMAND ----------

# DBTITLE 1,View deployments
deployments = rag_studio.list_deployments()
for deployment in deployments:
  if deployment.model_name == uc_model_fqdn and deployment.model_version==uc_registered_chain_info.version:
    print(parse_deployment_info(deployment))
