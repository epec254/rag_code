# RAG Studio Demo Code

Note: This feature is in [Private Preview](https://docs.databricks.com/en/release-notes/release-types.html). To try it, reach out to your Databricks contact or [rag-feedback@databricks.com](mailto:rag-feedback@databricks.com).

The software and other materials included in this repo ("Copyrighted Materials") are protected by US and international copyright laws and are the property of Databricks, Inc. The Copyrighted Materials are not provided under a license for public or third-party use. Accordingly, you may not access, use, copy, modify, publish, and/or distribute the Copyrighted Materials unless you have received prior written authorization or a license from Databricks to do so.

# What is included?

**RAG Studio:** A marketing name that encompasses the upgraded Mosaic AI platform capabilities for building high-quality Retrieval Augmented Generation (RAG) applications:
  - **MLflow:** Support for logging, parameterizing, and tracing Chains that are unified between development & production.  Chains can be logged as code vs. pickled.
  - **Model Serving:** Support for hosting Chains e.g., token streaming, automated authentication of Databricks services used in your chain, feedback API and a simplified chain deployment API
  - **RAG Cookbook:** Sample code & how-to guide offering an opinionated end-to-end workflow for building RAG apps [this repo]
  - *[Future release] Lakehouse Monitoring: Capabilities for monitoring your apps once in production*

**Evaluation Suite:** Built-for-purpose tools to evaluate Generative AI Apps, starting with RAG apps:
  - **Evaluation Harness:** evaluate(...) command that runs the evaluation
  - **Review App:** UI tool for collecting stakeholder feedback & building evaluation sets
  - **Databricks LLM Judges:** Databricks' proprietary AI-assisted judges for evaluating RAG quality.  Can be tuned with customer provided examples to increase agreement with human raters.
  - **Metrics:** A set of Databricks-defined metrics for measuring quality/cost/latency of your chain.  Most metrics are defined using the output of the Databricks LLM judges.
  - **Customer-defined LLM Judges:** Databricks framework to quickly define custom judges that evaluate business / use-case specific aspects of quality
  - *[Future release] Custom metrics: Provide a user-defined function to run and record its value as an evaluation metric.*


# Documentation

Our documentation provides a comprehensive overview of the above functionality:
- [RAG Studio & Evaluation Suite.pdf](<RAG Studio Overview Docs.pdf>)
- [MLflow Tracing.pdf](<Documentation/MLflow Tracing.pdf>)


# Sample code

*Note: While stored in the Git repo as `.py` files, these `.py` files are actually Databricks Notebooks - if you import the file using Databricks, it will render as a Notebook in the Notebook editor.  **To get started, clone this repo as a Git Folder in your Databricks workspace.***

## RAG Cookbook

## Tutorials: Evaluation Suite

### Running evaluation on an existing RAG chain / app

### Coming soon
These items are currently covered in the documentation, but we will release additional hands-on tutorials.  If you need one of these sooner, please contact us at [rag-feedback@databricks.com](mailto:rag-feedback@databricks.com).

- Improving LLM judge agreement with human raters using few-shot examples
- Curating an Evaluation Set using feedback from the Review App
- Measuring use-case specific aspects of quality with customer-defined LLM judges

## Tutorials: RAG Studio

Tutorials 1 and 2 introduce you to the basics of creating, logging, and parameterizing a Chain using MLflow's upgraded capabilities.

## Tutorial 1: Hello world

### Creating a chain
First, let's create the most minimal chain in RAG Studio.  This tutorial will familarize you with the basics of the development workflow and the chain input/output schema.

When using RAG Studio, you create your chain code using LangChain inside a Notebook.  Open the [`hello_world_chain`](M1_Sample_Code/1_hello_world/1_hello_world_chain.py) Notebook, review the commented code, and try running it.

Note: If you need support for Python based chains, please reach out to our team.

### Logging & deploying a chain

Now, let's log this chain to MLflow and deploy it to the Review App.

When using RAG Studio, you create, evaluate, and deploy new versions of your chain using a "Driver" notebook.  For now, let's just log and deploy the chain.  Open the [`1_hello_world_driver_notebook`](M1_Sample_Code/1_hello_world/1_hello_world_driver_notebook.py) Notebook, review the code, and try running it.

If successful, you will be able to chat with your hello world chain in the Review App.

## Tutorial 2: Hello World w/ parameterization

### Parameterizing a chain

RAG Studio supports parameterizing your chains - this allows you to quickly iterate on quality related parameters (such as the prompt or retriever configuruation) while holding the chain's code constant.

Parameterization is based on a YAML file.  Open [`2_hello_world_config.yaml`](M1_Sample_Code/2_hello_world_parameterized/2_hello_world_config.yaml) for an example of the configuration file - the structure of the YAML is up to you, although in the next tutorial, you will see our suggested structure.

Open [`2_hello_world_parameterized_chain`](M1_Sample_Code/2_hello_world_parameterized/2_hello_world_parameterized_chain.py) and [`2_hello_world_parameterized_driver_notebook`](M1_Sample_Code/2_hello_world_parameterized/2_hello_world_parameterized_driver_notebook.py) to see how this works in practice.


How to access configuration settings:
```
############
# Get the configuration YAML
############
model_config = mlflow.models.ModelConfig(development_config="1_hello_world_config.yaml")
model_config.get('sample_param')
```

## Tutorial 3: Creating & evaluating a RAG chain

NOw, we will create a simple RAG chain with PDF files from a UC Volume.  

### Create a UC Volume with PDF files

Create a UC Volume and load PDF files.  You can use the PDFs from the directory [`sample_pdfs`](M1_Sample_Code/sample_pdfs/) to get started quickly - these PDFs are a few recent research papers from Matei's lab.

### Create a Vector Index

Open and follow the steps in the notebook [`3_load_pdf_to_vector_index`](M1_Sample_Code/3_pdf_rag/3_load_pdf_to_vector_index.py) to load PDF files from the UC Volume to a Vector Index.  The sample notebook uses the BGE embedding model hosted on FMAPI, but later tutorials show you how to use OpenAI's embedding models.

### Prototype a RAG Chain

1. Take the output from the last cell in the [`3_load_pdf_to_vector_index`](M1_Sample_Code/3_pdf_rag/3_load_pdf_to_vector_index.py) notebook and overwrite the first few lines of the [`3_rag_chain_config.yaml`](M1_Sample_Code/3_pdf_rag/3_rag_chain_config.yaml) configuration so your chain can use the vector index you just created.

2. Open the notebook [`3_rag_chain`](M1_Sample_Code/3_pdf_rag/3_rag_chain.py) and run the code locally to test the chain.  This chain uses the `Databricks-DBRX-Instruct` model hosted on FMAPI.

### Log & evaluate a RAG Chain

To understand the evaluation metrics and LLM judges that are used to evaluate your chain, refer to the [metrics overview](metrics.md).

1. Open the notebook [`3_rag_chain_driver_notebook`](M1_Sample_Code/3_pdf_rag/3_rag_chain_driver_notebook.py) to log, evaluate, and deploy the chain.
2. Share the deployed Review App with your users to interact with the chain and provide feedback.

# Advanced examples & tutorials

## 4. Multi-turn conversation

The chain [`4_rag_chain_w_conversation_history`](M1_Sample_Code/4_rag_chain_w_conversation_history/4_rag_chain_w_conversation_history.py) and [`4_rag_chain_w_conversation_history_config.yaml`](M1_Sample_Code/4_rag_chain_w_conversation_history/4_rag_chain_w_conversation_history_config.yaml) is an example showing you how to enable multi-turn conversation with a query re-writer prompt.  The accompanying driver notebook [`4_rag_chain_w_conversation_history_driver_notebook`](M1_Sample_Code/4_rag_chain_w_conversation_history/4_rag_chain_w_conversation_history_driver_notebook.py) follows the same workflow as the driver notebook from [Tutorial 3](M1_Sample_Code/3_pdf_rag) to log and evaluate this chain.

