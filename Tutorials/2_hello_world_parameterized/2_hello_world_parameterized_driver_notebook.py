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

# Use Unity Catalog as the model registry
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the Unity Catalog location where the chain will be logged

# COMMAND ----------

# Create widgets 
dbutils.widgets.text("uc_catalog", "", "Unity Catalog")
dbutils.widgets.text("uc_schema", "", "Unity Catalog Schema")
dbutils.widgets.text("model_name", "hello_world_parameterized", "Model Name")

# Retrieve the values from the widgets
uc_catalog = dbutils.widgets.get("uc_catalog")
uc_schema = dbutils.widgets.get("uc_schema")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow logging input parameters

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
chain_notebook_file = "2_hello_world_parameterized_chain"
chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)

config_yaml_file = "configs/2_hello_world_config.yaml"
chain_yaml_path = os.path.join(os.getcwd(), config_yaml_file)

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Config notebook path: {chain_yaml_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Test different chain configurations to improve quality
# MAGIC
# MAGIC * We log the same chain, but with different configurations. This simulates iterating over different chain parameters and logging the resulting 
# MAGIC chains
# MAGIC * In this instance we log two different chains, with two different configs YAMLs to two separate MLflow runs. This will subsequently allow us to load these distinct chains from their respective MLflow runs and evaluate them.
# MAGIC
# MAGIC **NOTE:** The chains have not been deployed at this point. We are simply logging the chain artifacts to MLflow.

# COMMAND ----------

############
# This is a toy example, but you can use this approach to test various parameters - different prompts, foundation models, retrieval configurations, chunking settings, etc.
############

# Define the configurations to test
# The Dict in `config`, `{"sample_param": "do this thing"}`, mirrors the keys in the 2_hello_world_config.yaml file
# When we log the models, we will save a reference to the logged model in the `logged_chain_info` key.
configs_to_test = {
    "config_boring": {
        "config": {
            "prompt_template": "You are a hello world bot.  Respond with a reply to the user's question that is boring.  Always include ""I am boring"" in your response.  User's question: {question}",
            "prompt_template_input_vars": ["question"],
            "model_serving_endpoint": "databricks-dbrx-instruct",
            "llm_parameters": {"temperature": 0.01, "max_tokens": 500},
        }
    },
    "config_fun": {
        "config": {
            "prompt_template": "You are a hello world bot.  Respond with a reply to the user's question that is fun and interesting to the user.  Always include ""I am super fun"" in your response. User's question: {question}",
            "prompt_template_input_vars": ["question"],
            "model_serving_endpoint": "databricks-dbrx-instruct",
            "llm_parameters": {"temperature": 0.01, "max_tokens": 500},
        }
    },
    "config_yaml": {"config": chain_yaml_path},
}

# Log each configuration to an MLflow Run
# The parameters specified in the configuration are logged to the run
for config_name, config_details in configs_to_test.items():
    print(f"Config: {config_name}")

    with mlflow.start_run(run_name=config_name):
        logged_chain_info = mlflow.langchain.log_model(
            lc_model=chain_notebook_path,
            model_config=config_details[
                "config"
            ],  # The configuration to test - this can also be a YAML file path rather than a Dict.
            artifact_path="chain",
            input_example=input_example,
            example_no_conversion=True,  # required to allow the schema to work
        )
        # Save a pointer to the Run for later evaluation of the chains.
        config_details["logged_chain_info"] = logged_chain_info

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the chains
# MAGIC
# MAGIC * Within a notebook setting, we can load in the separate chains, and query them with `chain.invoke()`
# MAGIC * Here we iteratively load in the two separate logged chains, and query them with `model_input`
# MAGIC * This is a simplified approach for demonstration. In a practical setting we recommend using `evaluate()` to more robustly evaluate the chain

# COMMAND ----------

############
# Now, let's test both models locally
# In the actual usage, you would use rag.evaluate(...) to run LLM judge and evaluate each chain's quality/cost/latency.
############

model_input = {
    "messages": [
        {
            "role": "user",
            "content": "What is Databricks?",
        }
    ]
}

for config_name, config_items in configs_to_test.items():
    print(f"Config: {config_name}")
    chain = mlflow.langchain.load_model(config_items['logged_chain_info'].model_uri)
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
winning_config = "config_fun"
print("Winning config: ", configs_to_test[winning_config])

winning_model_uri = configs_to_test[winning_config]["logged_chain_info"].model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.

# COMMAND ----------

# Unity Catalog location
uc_model_fqn = f"{uc_catalog}.{uc_schema}.{model_name}"

# Register the model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(model_uri=winning_model_uri, name=uc_model_fqn)

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy the chain to:
# MAGIC 1. Review App so you & your stakeholders can chat with the chain & given feedback via a web UI.
# MAGIC 2. Chain REST API endpoint to call the chain from your front end.
# MAGIC 3. Feedback REST API endpoint to pass feedback back from your front end.
# MAGIC
# MAGIC **Note:** It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

deployment_info = agents.deploy(model_name=uc_model_fqn, model_version=uc_registered_model_info.version)

# Note: It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.
