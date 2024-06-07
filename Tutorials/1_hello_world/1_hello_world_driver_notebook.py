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
# MAGIC %pip install databricks-agents 'mlflow>=2.13'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../utilities/prpr_shared_funcs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

# DBTITLE 1,Imports
import os
import mlflow
from databricks import agents

# Use the Unity Catalog model registry
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the Unity Catalog location where the chain will be logged

# COMMAND ----------

# Create widgets 
dbutils.widgets.text("uc_catalog", "", "Unity Catalog")
dbutils.widgets.text("uc_schema", "", "Unity Catalog Schema")
dbutils.widgets.text("model_name", "hello_world", "Model Name")

# Retrieve the values from the widgets
uc_catalog = dbutils.widgets.get("uc_catalog")
uc_schema = dbutils.widgets.get("uc_schema")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the chain to MLflow
# MAGIC
# MAGIC Log the chain to the Notebook's MLflow Experiment inside a Run. The model is logged to the Notebook's MLflow Experiment as a run.

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow logging input parameters

# COMMAND ----------

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
chain_notebook_file = "1_hello_world_chain"
chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)

print(f"Chain notebook path: {chain_notebook_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the chain

# COMMAND ----------

with mlflow.start_run():
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,
        artifact_path="chain",
        input_example=input_example,
        example_no_conversion=True, # required to allow the schema to work
    )

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
# MAGIC

# COMMAND ----------

# DBTITLE 1,Deploy the model
deployment_info = agents.deploy(uc_model_fqn, uc_registered_model_info.version)

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
deployments = agents.list_deployments()
for deployment in deployments:
  if deployment.model_name == uc_model_fqn and deployment.model_version==uc_registered_model_info.version:
    print(parse_deployment_info(deployment))
