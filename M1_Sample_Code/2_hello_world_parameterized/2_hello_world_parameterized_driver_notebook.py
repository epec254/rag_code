# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 2. Hello World Parameterized Driver Notebook
# MAGIC
# MAGIC This notebook demonstrates how to use Databricks RAG Studio to log and test chains with different configurations. It covers the following steps:
# MAGIC
# MAGIC 1. Install required libraries and import required modules
# MAGIC 3. Define paths for the chain notebook and config YAML
# MAGIC 4. Log the chain to MLflow and test it locally
# MAGIC 5. Test different chain configurations to improve quality
# MAGIC 6. Log and evaluate the chain with different configurations

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
# MAGIC ## Define paths for chain notebook and config YAML

# COMMAND ----------

# DBTITLE 1,Setup
# Specify the full path to the chain notebook & config YAML
current_path = '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

chain_notebook_file = "2_hello_world_parameterized_chain"
chain_config_file = "2_hello_world_config.yaml"

chain_notebook_path = os.path.join(current_path, chain_notebook_file)
chain_config_path = os.path.join(current_path, "configs", chain_config_file)

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain config path: {chain_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the chain to MLflow and test locally
# MAGIC
# MAGIC Log the chain to the Notebook's MLflow Experiment inside a Run. The model is logged to the Notebook's MLflow Experiment as a run.
# MAGIC
# MAGIC **NOTE:** Here we log the chain as is. The config file for this chain is specified in the chain notebook

# COMMAND ----------

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
# MAGIC ## Test different chain configurations to improve quality
# MAGIC
# MAGIC * We log the same chain, but with different config files. This simulates iterating over different chain parameters and logging the resulting 
# MAGIC chains
# MAGIC * In this instance we log two different chains, with two different configs YAMLs to two separate MLflow runs. This will subsequently allow us to load these distinct chains from their respective MLflow runs and evaluate them.
# MAGIC
# MAGIC **NOTE:** The chains have not been deployed at this point. We are simply logging the chain artifacts to MLflow.

# COMMAND ----------

# DBTITLE 1,Example configs
client = mlflow.tracking.MlflowClient()

############
# This is a toy example, but you can use this approach to test various parameters - different prompts, foundation models, retrieval configurations, chunking settings, etc.
############

# This JSON structure `{"sample_param": "do this thing"}` mirrors the keys in the 2_hello_world_config.yaml file
configs_to_test = [
    {"name": "config_1", "config": {"sample_param": "do this thing"}},
    {"name": "config_2", "config": {"sample_param": "do a different thing"}},
]

# Create dict of configs
# This JSON structure `{"sample_param": "do this thing"}` mirrors the keys in the 2_hello_world_config.yaml file
configs_to_test = {
    "config_1": {"sample_param": "do this thing"},
    "config_2": {"sample_param": "do a different thing"}
}

# Log each configuration to an MLflow Run
for config_name, config in configs_to_test.items():
    print(f"Config: {config_name}")
    
    # Write the config to a YAML
    yaml_file = f"2_hello_world_config_{config_name}.yaml"
    yaml_path = os.path.join(current_path, "configs", yaml_file)
    with open(yaml_path, "w") as file:
        yaml.dump(config, file)  
    print(yaml.dump(config))
    
    # Log the chain w/ the config to an MLflow Run in the Notebook's Experiment
    logged_chain_info = rag_studio.log_model(
        code_path=chain_notebook_path, config_path=yaml_path
    )
    # Tag the MLflow run
    client.set_tag(logged_chain_info.run_id, "config_name", config_name)
    
    configs_to_test[config_name]["logged_chain_info"] = logged_chain_info
    
    print(f"MLflow Run: {logged_chain_info.run_id}")
    print(f"Model URI: {logged_chain_info.model_uri}")
    print("--\n")    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the chains
# MAGIC
# MAGIC * Within a notebook setting, we can load in the separate chains, and query them with `chain.invoke()`
# MAGIC * Here we iteratively load in the two separate logged chains, and query them with `model_input`
# MAGIC * This is a simplified approach for demonstration. In a practical setting we recommend using `rag.evaluate()` to more robustly evaluate the chain

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

for config_name in configs_to_test:
    print(f"Config: {config_name}")
    config = configs_to_test[config_name]
    chain = mlflow.langchain.load_model(config['logged_chain_info'].model_uri)
    print(chain.invoke(model_input))
    print("--\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy winning chain
# MAGIC
# MAGIC * Once the best performing chain has been determined, we can deploy that chain.
# MAGIC * For the purpose of demonstration, let's assume that we have found `config_2` to yield the highest quality chain

# COMMAND ----------

# Define winning config 
winning_config = "config_2"
print("Winning config: ", configs_to_test[winning_config])

winning_model_uri = configs_to_test[winning_config]["logged_chain_info"].model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.

# COMMAND ----------

# Change these values to your catalog and schema
uc_catalog = "niall_dev"
uc_schema = "rag"
model_name = "hello_world_2"
uc_model_fqdn = f"{uc_catalog}.{uc_schema}.{model_name}" 

uc_registered_chain_info = mlflow.register_model(winning_model_uri, uc_model_fqdn)

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy the chain to:
# MAGIC 1. Review App so you & your stakeholders can chat with the chain & given feedback via a web UI.
# MAGIC 2. Chain REST API endpoint to call the chain from your front end.
# MAGIC 3. Feedback REST API endpoint to pass feedback back from your front end.
# MAGIC
# MAGIC **Note:** It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

deployment_info = rag_studio.deploy_model(uc_model_fqdn, uc_registered_chain_info.version)
print(parse_deployment_info(deployment_info))

# Note: It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.
