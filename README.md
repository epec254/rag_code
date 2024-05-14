# RAG Studio Demo Code

Note: This feature is in [Private Preview](https://docs.databricks.com/en/release-notes/release-types.html). To try it, reach out to your Databricks contact or [rag-feedback@databricks.com](mailto:rag-feedback@databricks.com).

The software and other materials included in this repo ("Copyrighted Materials") are protected by US and international copyright laws and are the property of Databricks, Inc. The Copyrighted Materials are not provided under a license for public or third-party use. Accordingly, you may not access, use, copy, modify, publish, and/or distribute the Copyrighted Materials unless you have received prior written authorization or a license from Databricks to do so.

# Overview of product names & features

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

# Table of contents

1. [Product documentation](#Product documentation)
2. [Known limitations](#Known limitations)
3. [Sample code](#Sample code)
    - [RAG Cookbook](#RAG Cookbook)
      - [PDF Bot w/ single-turn conversation](#PDF Bot w/ single-turn conversation)
      - [PDF Bot w/ multi-turn conversation](#PDF Bot w/ multi-turn conversation)
      - [Advanced data pipeline for tuning parsing, chunking, embedding strategy](#Advanced data pipeline for tuning parsing, chunking, embedding strategy)
    - [How to tutorials](#How to tutorials)
      - [Tutorial 1: Creating, logging & deploying chains](#Tutorial 1: Creating, logging & deploying chains)
      - [Tutorial 2: Parameterizing chains](#Tutorial 2: Parameterizing chains)
      - [Tutorial 3: Running evaluation on a logged RAG chain](#Tutorial 3: Running evaluation on a logged RAG chain)
      - [Tutorial 4: Running evaluation on a RAG chain or app built outside of RAG Studio](#Tutorial 4: Running evaluation on a RAG chain or app built outside of RAG Studio)

# Product documentation

Our documentation provides a comprehensive overview of the above functionality:
- [RAG Studio & Evaluation Suite.pdf](<RAG Studio Overview Docs.pdf>)
- [MLflow Tracing.pdf](<Documentation/MLflow Tracing.pdf>)

# Known limitations

- Only tested on Databricks Runtime 15.0 and 14.3 Single User clusters.  They have not been tested on MLR or Shared Clusters.
- Only supports chains using the Langchain framework. Generic Python functionality is coming soon.
- Chains that need custom credentials for external services e.g., directly calling 3rd party APIs require these credentials to be manually configured in the model serving UI after calling `deploy_model(...)`
- Support for custom Python library dependencies and versions e.g., `pip_requirements` in `mlflow.langchain.log_model(...)` has not been tested extensively.
- Serialization-based MLFlow logging has not been tested with RAG Studio
- Code-based MLflow logging captures all loaded Python packages in the Driver Notebook as the `pip_requirements` for the MLflow model - if you need to add or remove requirements, pass a custom `pip_requirements` array that includes `"databricks-rag-studio==0.2.0"`.
- Some parts of the product documentation are still work-in-progress



# Sample code

*Note: While stored in the Git repo as `.py` files, these `.py` files are actually Databricks Notebooks - if you import the file using Databricks, it will render as a Notebook in the Notebook editor.  **To get started, clone this repo as a Git Folder in your Databricks workspace.***

## RAG Cookbook

### PDF Bot w/ single-turn conversation

[This cookbook](RAG Cookbook/3_pdf_rag_with_single_turn_chat/README.md) creates a simple RAG chain with PDF files stored in a UC Volume.  

### PDF Bot w/ multi-turn conversation

[This cookbook](RAG Cookbook/4_pdf_rag_with_multi_turn_chat/README.md) creates a multi-turn conversation capable RAG chain with PDF files stored in a UC Volume.  *This cookbook is identical to the single-turn converastion cookbook, except for mintor changes to the chain & configuration to support multi-turn conversations.*


### Advanced data pipeline for tuning parsing, chunking, embedding strategy

## How to tutorials

### Tutorial 1: Creating, logging & deploying chains

[This tutorial](Tutorials/1_hello_world/README.md) walks you through how to create, log and deploy a chain.  The outcome is a user-facing Web UI for chatting with the chain & providing feedback (the Review App) and a REST API for integrating the chain into your user-facing application.

### Tutorial 2: Parameterizing chains

[This tutorial](Tutorials/2_hello_world_parameterized/README.md) walks you through RAG Studio's support for parameterizing chains.  Parameterization allows you to quickly iterate on quality related parameters (such as the prompt or retriever configuruation) while holding the chain's code constant.

### Tutorial 3: Running evaluation on a logged RAG chain

[This tutorial](Tutorials/3_evaluation_of_existing_chains/README.md) walks you through using Evaluation Suite to evaluate the quality of a RAG chain built with RAG Studio.

### Tutorial 4: Running evaluation on a RAG chain or app built outside of RAG Studio

[This tutorial](Tutorials/4_evaluation_of_existing_chains/README.md) walks you through using Evaluation Suite to evaluate the quality of a RAG chain built outside of RAG Studio or already deployed outside of Databricks.

### Additional tutorials coming soon
These items are currently covered in the documentation, but will be covered in future hands-on tutorials.  If you need one of these sooner, please contact us at [rag-feedback@databricks.com](mailto:rag-feedback@databricks.com).

- Improving LLM judge agreement with human raters using few-shot examples
- Curating an Evaluation Set using feedback from the Review App
- Measuring use-case specific aspects of quality with customer-defined LLM judges
