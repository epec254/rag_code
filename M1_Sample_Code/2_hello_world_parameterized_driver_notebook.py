# Databricks notebook source
# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %run ./wheel_installer

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

# DBTITLE 1,Imports
import os
import mlflow
import yaml
from databricks import rag_studio

############
# RAG Studio dependes on MLflow to show a trace of your chain. The trace can help you easily debug your chain and keep track of inputs & responses your chain performs well or performs poorly.
############
mlflow.langchain.autolog()

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

# DBTITLE 1,Setup
############
# Specify the full path to the chain notebook & config YAML
############

# Assuming your chain notebook is in the current directory, this helper line grabs the current path, prepending /Workspace/
# Limitation: RAG Studio does not support logging chains stored in Repos
current_path = '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

chain_notebook_file = "2_hello_world_parameterized_chain"
chain_config_file = "2_hello_world_config.yaml"
chain_notebook_path = f"{current_path}/{chain_notebook_file}"
chain_config_path = f"{current_path}/{chain_config_file}"

print(f"Saving chain from: {chain_notebook_path}, config from: {chain_config_path}")

# COMMAND ----------

# MAGIC %md ## First, let's test the model as-is

# COMMAND ----------

############
# Log the chain to the Notebook's MLflow Experiment inside a Run
# The model is logged to the Notebook's MLflow Experiment as a run
############

logged_chain_info = rag_studio.log_model(code_path=chain_notebook_path)

print(f"MLflow Run: {logged_chain_info.run_id}")
print(f"Model URI: {logged_chain_info.model_uri}")

############
# If you see this error, go to your chain code and comment out all usage of `dbutils`
############
# ValueError: The file specified by 'code_path' uses 'dbutils' command which are not supported in a chain model. To ensure your code functions correctly, remove or comment out usage of 'dbutils' command.

# COMMAND ----------

############
# Test the model locally
# Note the `Config: this is a test.` portion of the output.
############
example_input = {
    "messages": [
        {
            "role": "user",
            "content": "Hello.",
        }
    ]
}

model = mlflow.langchain.load_model(logged_chain_info.model_uri)
model.invoke(example_input)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Use configuration to test various parameters to improve quality
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Example configs
client = mlflow.tracking.MlflowClient()

############
# This is a toy example, but you can use this approach to test various parameters - different prompts, foundational models, retrieval configurations, chunking settings, etc.
############

# This JSON structure `{"sample_param": "do this thing"}` mirrors the keys in the 2_hello_world_config.yaml file
configs_to_test = [
    {"name": "config_1", "config": {"sample_param": "do this thing"}},
    {"name": "config_2", "config": {"sample_param": "do a different thing"}},
]

# Log each configuration to an MLflow Run
for config_to_test in configs_to_test:
    print(f"Config: {config_to_test['name']}")
    # write the config to a YAML
    yaml_string = yaml.dump(config_to_test["config"])
    print(yaml_string)
    yaml_file = f"2_hello_world_config_{config_to_test['name']}.yaml"
    with open(yaml_file, "w") as file:
        file.write(yaml_string)
    yaml_path = f"{current_path}/{yaml_file}"

    # Log the chain w/ the config to an MLflow Run in the Notebook's Experiment
    logged_chain_info = rag_studio.log_model(
        code_path=chain_notebook_path, config_path=yaml_path
    )
    # Tag the MLflow run
    client.set_tag(logged_chain_info.run_id, "config_name", config_to_test['name'])
    print(f"MLflow Run: {logged_chain_info.run_id}")
    print(f"Model URI: {logged_chain_info.model_uri}")
    config_to_test["logged_chain_info"] = logged_chain_info
    print("--\n")

# COMMAND ----------

############
# Now, let's test both models locally
# In the actual usage, you would use rag.evaluate(...) to run LLM judge and evaluate each chain's quality/cost/latency.
############

model_input = {
    "messages": [
        {
            "role": "user",
            "content": "Hello world!!",
        }
    ]
}

for config_to_test in configs_to_test:
    print(f"Config: {config_to_test['name']}")
    chain = mlflow.langchain.load_model(config_to_test['logged_chain_info'].model_uri)
    print(chain.invoke(model_input))
    print("--\n")

