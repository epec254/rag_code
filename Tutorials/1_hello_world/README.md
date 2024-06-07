# Tutorial 1: Creating, logging & deploying chains

Logging a chain is the basis of your development process.  Logging captures a "point in time" of the chain's code and configuration so you can evaluate the quality of those settings.  With RAG Studio, there are 2 ways you can create & log a chain, both based on MLflow:

|Approach|Description|Pros|Cons|
|-------|-------|-------|-------|
|**Serialization-based MLflow logging**|The chain's code and current state in the Python environment is [serialized](https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/part1-named-flavors.html) to disk, often using libraries such as `pickle` or `joblib`.  The Python environment (e.g., pip packages) is captured as a list of packages.  <br/><br/>When the chain is deployed, the Python environment is restored, and the serialized object is loaded into memory so it can be invoked when the endpoint is called.|+ `log_model(...)` can be called from the same notebook as where the model is defined| - Original code is not available<br/>- All used libraries/objects must support serialization. |
|**Code-based MLflow logging**| The chain's code is captured as a Python file.  The Python environment (e.g., pip packages) is captured as a list of packages.  <br/><br/>When the chain is deployed, the Python environment is restored, and the chain's code is executed to load the chain into memory so it can be invoked when the endpoint is called. | + Overcomes inherent limitations of serialization - it is not supported by many popular GenAI libraries. <br/> + Saves a copy of the original code for later reference <br/>+ No need to restructure your code into a single object that can be serialized. <br/> | - `log_model(...)` must be called from a *different* notebook than the chain's code (called a Driver Notebook).<br/>- *(current limitation)* The Python environment of the Driver Notebook is captured when logging - if this is different than the environment required by the chain's code, you must manually specicify the Python packages when logging  |

For developing RAG applications, Databricks suggests approach (2) code-based logging.  RAG Studio's documentation assumes you are using code-based logging & has not been tested with serialization-based logging.

#### **1. Creating a chain**

In RAG Studio, the chain's code, which we colloquially refer to as `chain.py`, is your development space - you can use any package, library, or technique.  The *only* requirement is that your chain comply with a given input / output schema.  Why?  To use your chain in the Review App & to evaluate your chain with Evaluation Suite, we must know how to send your chain user queries & how to find the chain's response back to the user.

Open the [`1_hello_world_chain`](Tutorials/1_hello_world/1_hello_world_chain.py) to see an example `chain.py`.  Review the code, and try running it!

*Note: In the next release of RAG Studio, we will allow for more flexible input / output schemas.*
<!--
##### Input schema
Your chain must accept an array of [OpenAI-formatted messages](https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chatmessage) as a `messages` parameter.
```
# This is the same input your chain's REST API will accept.
question = {
    "messages": [
        {
            "role": "user",
            "content": "question 1",
        },
        {
            "role": "assistant",
            "content": "answer 1",
        },
        {
            "role": "user",
            "content": "new question!!",
        },
    ]
}
```

##### Output schema
Your chain must return a single string value.  To do this in LangChain, use `StrOutputParser()` as your final chain step.
```
chain = (
    {
        "user_query": itemgetter("messages")
        | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
    }
    | RunnableLambda(fake_model)
    | StrOutputParser()
)
```
-->
#### 2. Logging the chain

When using code-based MLflow logging, you log, evaulate, and deploy new versions of your `chain.py` using a seperate notebook, which we call the Driver Notebook.  

[`1_hello_world_driver_notebook`](Tutorials/1_hello_world/1_hello_world_driver_notebook.py) is a sample Driver Notebook that contains this step (logging) and the next step (deploying).

```python
import mlflow 

code_path = "/Workspace/Users/first.last/chain.py"
config_path = "/Workspace/Users/first.last/config.yml"

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is Retrieval-augmented Generation?",
        }
    ]
}

with mlflow.start_run():
  logged_chain_info = mlflow.langchain.log_model(
    lc_model=code_path, # Passing a file path here tells `log_model(...)` to use code-based logging vs. serialization-based logging.
    model_config=config_path, # Optional configuration of the chain - either a Python dictionary or path to a YAML file
    artifact_path="chain", # Any user provided string, used as the path inside the MLflow model where artifacts are stored
    input_example=input_example, # Must be a valid input to your chain 
    example_no_conversion=True, # Required to enable 
    extra_pip_requirements = ["databricks-rag-studio==0.2.0"] # Required during PrPr to tell Databricks, this is a RAG Studio compatible model
  )

print(f"MLflow Run: {logged_chain_info.run_id}")
print(f"Model URI: {logged_chain_info.model_uri}")

# model_uri can also be used to do evaluations of your RAG
model_uri=logged_chain_info.model_uri
```

#### Step 3. Evaluating the chain

Normally, at this point in your dev loop, you would evaluate the chain.  We will skip this step for simplicity of this tutorial.

#### Step 4. Deploying the chain to the Review App & a REST API

RAG Studio uses Databricks Model Serving to deploy your chain.  During development, you deploy your chain to collect feedback from expert stakeholders.  During production, you deploy your chain to make it available as a REST API that can be integrated into your user-facing application.  With RAG Studio, a single  `deploy(...)` command creates a scalable, production-ready deployment that works for either of these use cases.

**Before you deploy your model, you must register your logged model (from step 2) to the Unity Catalog:**

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")

catalog_name = "test_catalog"
schema_name = "schema"
model_name = "chain_name"

model_fqn = catalog_name + "." + schema_name + "." + model_name
uc_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=model_fqn)
```

**Then, you can deploy your model:**

```python
from databricks.agents import deploy
from mlflow.utils import databricks_utils as du

deployment = deploy(model_fqn, uc_model_info.version)

# query_endpoint is the URL that can be used to make queries to the app
deployment.query_endpoint

# Copy deployment.rag_app_url to browser and start interacting with your RAG application.
deployment.rag_app_url

# Temporary helper function during PrPr to pretty print URLs
def parse_deployment_info(deployment_info):
  browser_url = du.get_browser_hostname()
  message = f"""Deployment of {deployment_info.model_name} version {deployment_info.model_version} initiated.  This can take up to 15 minutes and the Review App & REST API will not work until this deployment finishes. 

  View status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}
  Review App: {deployment_info.rag_app_url}"""
  return message

parse_deployment_info(deployment)

```

**Calling `deploy(...)` does the following:**

1. Enables the Review App for your chain
    - Allows your expert stakeholders can chat with the chain & give feedback via a web UI
2. Creates REST APIs for your chain that can be integrated into your user-facing application
    - `invoke` endpoint to get responses from the chain
    - `feedback` to pass feedback from your front end
3. Complete logging of every request to the Review App or REST API e.g., input/output and intermediate trace via Inference Tables
    - 3 Delta Tables are created for each deployment
      1. Raw JSON payloads `{catalog_name}.{schema_name}.{model_name}_payload`
      2. Formatted request/response & MLflow Traces `{catalog_name}.{schema_name}.{model_name}_payload_request_logs`
      3. Formatted feedback, as provided in the Review App or via Feedback API, for each request `{catalog_name}.{schema_name}.{model_name}_payload_assessment_logs`

**Note:** It can take up to 15 minutes to deploy.  Raw JSON payloads take 10 - 30 minutes to arrive, and the formatted logs are processed from the raw payloads every ~hour.
