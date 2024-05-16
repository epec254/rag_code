### Tutorial 2: Parameterizing chains
RAG Studio supports parameterizing your chains - this allows you to quickly iterate on quality related parameters (such as the prompt or retriever configuruation) while holding the chain's code constant.

Parameterization is flexible and works with any key:value pairs you define.  In the RAG Cookbook, we provide a suggested set of key:value pairs.  Parameters can be stored inside a YAML file or a Python Dictionary.  Databricks suggests using a Dictionary during development and then converting your Dictionary to a YAML for production deployment via CI/CD.

Example files:
- [`2_hello_world_config.yaml`](Tutorials/2_hello_world_parameterized/2_hello_world_config.yaml) is an example of the YAML configuration file.
- [`2_hello_world_parameterized_chain`](Tutorials/2_hello_world_parameterized/2_hello_world_parameterized_chain.py) is an example of a parameterized chain 
- [`2_hello_world_parameterized_driver_notebook`](Tutorials/2_hello_world_parameterized/2_hello_world_parameterized_driver_notebook.py) is an example of a driver notebook to iterate on different parameters

**How to parameterize your chain**
1. Define a key:value parameter in a Dictionary or YAML file
    ```python
    config_dict = {"sample_param": "this is the sample parameter that can be changed! this could be a prompt, a retrieval setting, or ..."}
    ```
    ```yaml
    sample_param: "Hello from the YAML config!"
    ```
2. In your `chain.py`, add 1 of the following 2 lines of code to access your parameters.  
    ```python
    import mlflow
    model_config = mlflow.models.ModelConfig(development_config=config_dict)
    model_config = mlflow.models.ModelConfig(development_config='path_to_yaml_file.yaml')
    ```
3. `model_config` is a Python Dictionary with your parameters.
    ```python
    value = model_config.get('sample_param')
    ```

**Iterating on parameters**

1. Open your driver notebook where you have your model logging code.
2. When calling `mlflow.langchain.log_model(...)`, pass the configuration you wish to test to the `model_config` parameter.  You can pass a Dictionary object or path to a YAML file.
    ```python
    with mlflow.start_run():
      logged_chain_info = mlflow.langchain.log_model(
          lc_model=chain_notebook_path, # path to your chain.py
          ## look down
          model_config=...,  # Dictionary object or path to a YAML
          ## look up ^^
          artifact_path="chain",
          input_example=input_example,
          example_no_conversion=True,  # required to allow the schema to work
          extra_pip_requirements=[  # temporary workaround needed during Private Preview
              "databricks-rag-studio==0.2.0"
          ],
      )
    ```
3. The resulting MLflow model uses the configuration that was passed when logging.  The `development_config` is automatically overriden.
