## Cookbook: PDF Bot w/ single-turn conversation

This cookbook creates a simple RAG chain with PDF files stored in a UC Volume.  

### Create a UC Volume with PDF files

Create a UC Volume and load PDF files.  You can use the PDFs from the directory [`sample_pdfs`](../sample_pdfs/) to get started quickly - these PDFs are a few recent research papers from Matei's lab.

### Create a Vector Index

Open and follow the steps in the notebook [`1_load_pdf_to_vector_index`](1_load_pdf_to_vector_index.py) to load PDF files from the UC Volume to a Vector Index.  The sample notebook uses the BGE embedding model hosted on FMAPI.

### Prototype a RAG Chain

1. Open the notebook [`2_rag_chain`](2_rag_chain.py) and run the code locally to test the chain.  This chain uses the `Databricks-DBRX-Instruct` model hosted on FMAPI.

### Log, evaluate, deploy a RAG Chain

1. Take the JSON output from the last cell in the [`1_load_pdf_to_vector_index`](1_load_pdf_to_vector_index.py) notebook and overwrite the `baseline_config` in Cell 17 of [`3_single_turn_pdf_driver_notebook.py`](3_single_turn_pdf_driver_notebook.py) configuration so your chain will be logged with the configuration of the vector index you just created.

    - Pro tip: Modify the prompts here to try improving the quality of the RAG chain!

2. Run all cells in this notebook to log, evaluate, and deploy this chain!

3. Share the deployed Review App with your users to interact with the chain and provide feedback.