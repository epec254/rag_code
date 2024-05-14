# Databricks notebook source
# MAGIC %md
# MAGIC # Example PDF Parsing Pipeline
# MAGIC
# MAGIC This is an example notebook that provides a **starting point** to build a data pipeline that loads, parses, chunks, and embeds PDF files from a UC Volume into a Databricks Vector Search Index.  
# MAGIC
# MAGIC Getting the right parsing and chunk size requires iteration and a working knowledge of your data - this pipeline is easy to adapt and tweak in order to add more advanced logic.
# MAGIC
# MAGIC **Limitations:**
# MAGIC - This pipeline resets the index every time, mirroring the index to the files in the UC Volume.  
# MAGIC - Splitting based on tokens requires a cluster with internet access.  If you do not have internet access on your cluster, adjust the gold parsing step.
# MAGIC - You can't change column names in the Vector Index after the tables are initially created - to change column names, delete the Vector Index and re-sync.

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting Started
# MAGIC
# MAGIC 1. To get started, `Run All`.  
# MAGIC 2. You will be alerted to any configuration settings you need to config or issues you need to resolve.  
# MAGIC 3. After you resolve an issue or set a configuration setting, press `Run All` again to verify your changes.  
# MAGIC 4. Repeat until you don't get errors and press `Run All` a final time to execute the pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install libraries & import packages

# COMMAND ----------

# MAGIC %pip install -U --quiet pypdf==4.1.0 databricks-sdk langchain==0.1.13 tokenizers torch transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datetime import timedelta
from typing import List
import yaml
import warnings

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointStatusState,
    EndpointType,
    PipelineType,
    VectorIndexType,
)
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pyspark.sql import Column
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pypdf import PdfReader

import io

# Init workspace client
w = WorkspaceClient()

# Use optimizations if available
dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
if dbr_majorversion >= 14:
  spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Required configuration
# MAGIC
# MAGIC 1. Select a Vector Search endpoint
# MAGIC
# MAGIC If you do not have a Databricks Vector Search endpoint, follow these [steps](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) to create one.
# MAGIC
# MAGIC 2. Select UC Catalog, Schema, and UC Volume w/ PDF files.
# MAGIC
# MAGIC Note: By default, the bronze/silver/gold Delta Tables with parsed chunks will land into this same UC Catalog/Schema.  You can change this behavior below.

# COMMAND ----------

vector_search_endpoints_in_workspace = [item.name for item in w.vector_search_endpoints.list_endpoints() if item.endpoint_status.state == EndpointStatusState.ONLINE]

if len(vector_search_endpoints_in_workspace) == 0:
    raise Exception("No Vector Search Endpoints are online in this workspace.  Please follow the instructions here to create a Vector Search endpoint: https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint")

# Vector Search Endpoint Widget
if len(vector_search_endpoints_in_workspace) > 1024:  # use text widget if number of values > 1024
    dbutils.widgets.text(
        "vector_search_endpoint_name",
        defaultValue="",
        label="#1 VS endpoint",
    )
else:
    dbutils.widgets.dropdown(
        "vector_search_endpoint_name",
        defaultValue="",
        choices=vector_search_endpoints_in_workspace+[""],
        label="#1 Select VS endpoint",
    )
vector_search_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")

if vector_search_endpoint_name == '' or vector_search_endpoint_name is None:
    raise Exception("Please select a Vector Search endpoint to continue.")
else:
    print(f"Using `{vector_search_endpoint_name}` as the Vector Search endpoint.")

# UC Catalog widget
uc_catalogs = [row.catalog for row in spark.sql("SHOW CATALOGS").collect()]

if len(uc_catalogs) > 1024:  # use text widget if number of values > 1024
    dbutils.widgets.text(
        "uc_catalog_name",
        defaultValue="",
        label="#2 UC Catalog",
    )
else:
    dbutils.widgets.dropdown(
        "uc_catalog_name",
        defaultValue="",
        choices=uc_catalogs + [""],
        label="#2 Select UC Catalog",
    )
uc_catalog_name = dbutils.widgets.get("uc_catalog_name")

# UC Schema widget (Schema within the defined Catalog)
if uc_catalog_name != "" and uc_catalog_name is not None:
    spark.sql(f"USE CATALOG `{uc_catalog_name}`")
    uc_schemas = [row.databaseName for row in spark.sql(f"SHOW SCHEMAS").collect()]
    uc_schemas = [schema for schema in uc_schemas if schema != "__databricks_internal"]

    if len(uc_schemas) > 1024: # use text widget if number of values > 1024
        dbutils.widgets.text(
            "uc_schema_name",
            defaultValue="",
            label="#3 UC Schema",
        )
    else:
        dbutils.widgets.dropdown(
            "uc_schema_name",
            defaultValue="",
            choices=[""] + uc_schemas,
            label="#3 Select UC Schema",
        )
else:
    dbutils.widgets.dropdown(
        "uc_schema_name",
        defaultValue="",
        choices=[""],
        label="#3 Select UC Schema",
    )
uc_schema_name = dbutils.widgets.get("uc_schema_name")

# UC Volume widget (Volume within the defined Schema)
if uc_schema_name != "" and uc_schema_name is not None:
    spark.sql(f"USE CATALOG `{uc_catalog_name}`")
    spark.sql(f"USE SCHEMA `{uc_schema_name}`")
    uc_volumes = [row.volume_name for row in spark.sql(f"SHOW VOLUMES").collect()]

    if len(uc_volumes) > 1024:
        dbutils.widgets.text(
            "source_uc_volume",
            defaultValue="",
            label="#4 UC Volume w/ PDFs",
        )
    else:
        dbutils.widgets.dropdown(
            "source_uc_volume",
            defaultValue="",
            choices=[""] + uc_volumes,
            label="#4 Select UC Volume w/ PDFs",
        )
else:
    dbutils.widgets.dropdown(
        "source_uc_volume",
        defaultValue="",
        choices=[""],
        label="#4 Select UC Volume w/ PDFs",
    )

source_uc_volume = f"/Volumes/{uc_catalog_name}/{uc_schema_name}/{dbutils.widgets.get('source_uc_volume')}"

# Validation
if (uc_catalog_name == "" or uc_catalog_name is None) or (uc_schema_name == "" or uc_schema_name is None):
    raise Exception("Please select a UC Catalog & Schema to continue.")
else:
    print(f"Using `{uc_catalog_name}.{uc_schema_name}` as the UC Catalog / Schema.")

if source_uc_volume == "" or source_uc_volume is None:
    raise Exception("Please select a source UC Volume w/ PDF files to continue.")
else:
    print(f"Using {source_uc_volume} as the UC Volume Source.")

# COMMAND ----------

# MAGIC %md ## Optional: Configure parameters
# MAGIC
# MAGIC We suggest starting with the default values to verify the pipeline works end to end.  You'll need to tune these settings to optimize the retrieval quality for your data.
# MAGIC
# MAGIC When comparing multiple configurations (different chunking settings, embedding models, etc), we suggest adjusting the bronze/silver/gold names to indicate different versions.

# COMMAND ----------

# DBTITLE 1,Data Processing Workflow Manager
# Force this cell to re-run when these values are changed in the Notebook widgets
uc_catalog_name = dbutils.widgets.get("uc_catalog_name")
uc_schema_name = dbutils.widgets.get("uc_schema_name")
volume_raw_name = dbutils.widgets.get("source_uc_volume")

# Defaults
BGE_CONTEXT_WINDOW_LENGTH_TOKENS = 512
CHUNK_SIZE_TOKENS = 425
CHUNK_OVERLAP_TOKENS = 75
DATABRICKS_FMAPI_BGE_ENDPOINT = "databricks-bge-large-en"
FMAPI_EMBEDDINGS_TASK = "llm/v1/embeddings"

bronze_raw_files_table_name = (
    f"{uc_catalog_name}.{uc_schema_name}.bronze_{volume_raw_name}_raw"
)
silver_parsed_files_table_name = (
    f"{uc_catalog_name}.{uc_schema_name}.silver_{volume_raw_name}_parsed"
)
gold_chunks_table_name = (
    f"{uc_catalog_name}.{uc_schema_name}.gold_{volume_raw_name}_chunked"
)
gold_chunks_index_name = (
    f"{uc_catalog_name}.{uc_schema_name}.gold_{volume_raw_name}_chunked_index"
)

print(f"Bronze Delta Table w/ raw files: `{bronze_raw_files_table_name}`")
print(f"Silver Delta Table w/ parsed files: `{silver_parsed_files_table_name}`")
print(f"Gold Delta Table w/ chunked files: `{gold_chunks_table_name}`")
print(f"Vector Search Index mirror of Gold Delta Table: `{gold_chunks_index_name}`")
print("--")

dbutils.widgets.text(
    "embedding_endpoint_name",
    DATABRICKS_FMAPI_BGE_ENDPOINT,
    label="Parameter: embedding endpoint",
)
embedding_endpoint_name = dbutils.widgets.get("embedding_endpoint_name")

try:
    w.serving_endpoints.get(embedding_endpoint_name)
except ResourceDoesNotExist as e:
    error = f"Model serving endpoint {embedding_endpoint_name} does not exist."
    if embedding_endpoint_name == DATABRICKS_FMAPI_BGE_ENDPOINT:
        error = (
            error
            + " This is likely because FMAPI is not available in your region.  To deploy the BGE embedding model using FMAPI, please see: https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html#provisioned-throughput-serving-for-bge-model-notebook"
        )
    else:
        error = error + " Verify your endpoint is properly configured."
    raise Exception(error)

# if w.serving_endpoints.get(embedding_endpoint_name).task != FMAPI_EMBEDDINGS_TASK:
#     raise Exception(
#         f"Your endpoint `{embedding_endpoint_name}` is not of type {FMAPI_EMBEDDINGS_TASK}.  Visit the Foundational Model APIs documentation to create a compatible endpoint: https://docs.databricks.com/en/machine-learning/foundation-models/index.html"
#     )

print(f"Embedding model endpoint: `{embedding_endpoint_name}`")
print("--")
dbutils.widgets.text(
    "chunk_size_tokens", str(CHUNK_SIZE_TOKENS), label="Parameter: chunk size"
)
chunk_size_tokens = int(dbutils.widgets.get("chunk_size_tokens"))

dbutils.widgets.text(
    "chunk_overlap_tokens", str(CHUNK_OVERLAP_TOKENS), label="Parameter: chunk overlap"
)
chunk_overlap_tokens = int(dbutils.widgets.get("chunk_overlap_tokens"))

if (
    embedding_endpoint_name == DATABRICKS_FMAPI_BGE_ENDPOINT
    and (chunk_size_tokens + chunk_overlap_tokens) > BGE_CONTEXT_WINDOW_LENGTH_TOKENS
):
    print(
        f"WARNING: Your chunk configuration exceeds `{embedding_endpoint_name}` context window of {BGE_CONTEXT_WINDOW_LENGTH_TOKENS} tokens.  Embedding performance may be diminished since tokens past {BGE_CONTEXT_WINDOW_LENGTH_TOKENS} tokens are ignored by the embedding model."
    )
else:
    print(
        f"Using chunking parameters: chunk_size_tokens: {chunk_size_tokens}, chunk_overlap_tokens: {chunk_overlap_tokens}"
    )

# COMMAND ----------

# If you want to run this pipeline as a Job, remove the above 2 cells and uncomment this code.

# # Defaults
# BGE_CONTEXT_WINDOW_LENGTH_TOKENS = 512
# CHUNK_SIZE_TOKENS = 425
# CHUNK_OVERLAP_TOKENS = 75
# DATABRICKS_FMAPI_BGE_ENDPOINT = "databricks-bge-large-en"
# FMAPI_EMBEDDINGS_TASK = "llm/v1/embeddings"

# # Vector Search Endpoint
# dbutils.widgets.text(
#     "vector_search_endpoint_name",
#     defaultValue="",
#     label="#1 VS endpoint",
# )
# vector_search_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")
# print("--")
# print(f"Using `{vector_search_endpoint_name}` as the Vector Search endpoint.")

# # UC Catalog
# dbutils.widgets.text(
#     "uc_catalog_name",
#     defaultValue="catalog_name",
#     label="#2 UC Catalog",
# )
# uc_catalog_name = dbutils.widgets.get("uc_catalog_name")

# # UC Schema
# dbutils.widgets.text(
#     "uc_schema_name",
#     defaultValue="",
#     label="#3 UC Schema",
# )
# uc_schema_name = dbutils.widgets.get("uc_schema_name")
# print("--")
# print(f"Using `{uc_catalog_name}.{uc_schema_name}` as the UC Catalog / Schema.")

# # UC Volume
# dbutils.widgets.text(
#     "source_uc_volume",
#     defaultValue="volume_name",
#     label="#4 UC Volume w/ PDFs",
# )

# volume_raw_name = dbutils.widgets.get("source_uc_volume")

# source_uc_volume = f"/Volumes/{uc_catalog_name}/{uc_schema_name}/{dbutils.widgets.get('source_uc_volume')}"
# print("--")
# print(f"Using {source_uc_volume} as the UC Volume Source.")

# bronze_raw_files_table_name = (
#     f"{uc_catalog_name}.{uc_schema_name}.bronze_{volume_raw_name}_raw"
# )
# silver_parsed_files_table_name = (
#     f"{uc_catalog_name}.{uc_schema_name}.silver_{volume_raw_name}_parsed"
# )
# gold_chunks_table_name = (
#     f"{uc_catalog_name}.{uc_schema_name}.gold_{volume_raw_name}_chunked"
# )
# gold_chunks_index_name = (
#     f"{uc_catalog_name}.{uc_schema_name}.gold_{volume_raw_name}_chunked_index"
# )
# print("--")
# print(f"Bronze Delta Table w/ raw files: `{bronze_raw_files_table_name}`")
# print(f"Silver Delta Table w/ parsed files: `{silver_parsed_files_table_name}`")
# print(f"Gold Delta Table w/ chunked files: `{gold_chunks_table_name}`")
# print(f"Vector Search Index mirror of Gold Delta Table: `{gold_chunks_index_name}`")
# print("--")

# dbutils.widgets.text(
#     "embedding_endpoint_name",
#     DATABRICKS_FMAPI_BGE_ENDPOINT,
#     label="Parameter: embedding endpoint",
# )
# embedding_endpoint_name = dbutils.widgets.get("embedding_endpoint_name")

# print(f"Embedding model endpoint: `{embedding_endpoint_name}`")
# print("--")
# dbutils.widgets.text(
#     "chunk_size_tokens", str(CHUNK_SIZE_TOKENS), label="Parameter: chunk size"
# )
# chunk_size_tokens = int(dbutils.widgets.get("chunk_size_tokens"))

# dbutils.widgets.text(
#     "chunk_overlap_tokens", str(CHUNK_OVERLAP_TOKENS), label="Parameter: chunk overlap"
# )
# chunk_overlap_tokens = int(dbutils.widgets.get("chunk_overlap_tokens"))

# COMMAND ----------

# MAGIC %md # Pipeline code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze: Load the files from the UC Volume
# MAGIC
# MAGIC **NOTE:** You will have to upload some PDF files to this volume. See the `sample_pdfs` folder of this repo for some example PDFs to upload to the UC Volume.

# COMMAND ----------

# DBTITLE 1,Recursive PDF Ingestion Workflow
LOADER_DEFAULT_DOC_URI_COL_NAME = "path"
DOC_URI_COL_NAME = "doc_uri"

bronze_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.pdf")
    .load(source_uc_volume)
)

bronze_df = bronze_df.selectExpr(f"* except({LOADER_DEFAULT_DOC_URI_COL_NAME})", f"{LOADER_DEFAULT_DOC_URI_COL_NAME} as {DOC_URI_COL_NAME}")

bronze_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(bronze_raw_files_table_name)

# reload to get correct lineage in UC
bronze_df = spark.read.table(bronze_raw_files_table_name)

display(bronze_df.selectExpr(f"{DOC_URI_COL_NAME}", "modificationTime", "length"))

if bronze_df.count() == 0:
    url = f"https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/explore/data{source_uc_volume}/"
    display(f"`{source_uc_volume}` does not contain any PDF files.  Open the volume and upload at least 1 PDF file: {url}")
    raise Exception(f"`{source_uc_volume}` does not contain any PDF files.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: Parse the PDF files into text
# MAGIC
# MAGIC If you want to change the parsing library or adjust it's settings, modify the contents of the `parse_pdf` UDF.

# COMMAND ----------

# DBTITLE 1,Optimized PDF Parsing Function
# If using runtime < 14.3, remove `useArrow=True`
# useArrow=True which optimizes performance only works with 14.3+

@F.udf(
    returnType=StructType(
        [
            StructField("number_pages", IntegerType(), nullable=True),
            StructField("text", StringType(), nullable=True),
            StructField("status", StringType(), nullable=False),
        ]
    ),
    # useArrow=True, # set globally
)
def parse_pdf(pdf_raw_bytes):
    try:
        pdf = io.BytesIO(pdf_raw_bytes)
        reader = PdfReader(pdf)
        output_text = ""
        for _, page_content in enumerate(reader.pages):
            output_text += page_content.extract_text() + "\n\n"

        return {
            "number_pages": len(reader.pages),
            "text": output_text,
            "status": "SUCCESS",
        }
    except Exception as e:
        return {"number_pages": None, "text": None, "status": f"ERROR: {e}"}


# Run the parsing
df_parsed = bronze_df.withColumn("parsed_output", parse_pdf("content")).drop("content")

# Check and warn on any errors
num_errors = df_parsed.filter(F.col("parsed_output.status") != "SUCCESS").count()
if num_errors > 0:
    warning.warn(f"{num_errors} documents had parse errors.  Please review.")

df_parsed.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(silver_parsed_files_table_name)

# reload to get correct lineage in UC and to filter out any error rows for the downstream step.
df_parsed = spark.read.table(silver_parsed_files_table_name).filter(
    F.col("parsed_output.status") == "SUCCESS"
)

display(df_parsed)

# COMMAND ----------

# MAGIC %md ## Gold: Chunk the parsed text
# MAGIC
# MAGIC If you change your embedding model, you will need to adjust the tokenizer accordingly.
# MAGIC
# MAGIC If you are using a cluster without internet access, remove the below cell and replace the udf with
# MAGIC
# MAGIC ```
# MAGIC @F.udf(returnType=ArrayType(StringType()), useArrow=True)
# MAGIC def split_char_recursive(content: str) -> List[str]:
# MAGIC     text_splitter = RecursiveCharacterTextSplitter(
# MAGIC         chunk_size=chunk_size, chunk_overlap=chunk_overlap
# MAGIC     )
# MAGIC     chunks = text_splitter.split_text(content)
# MAGIC     return [doc for doc in chunks]
# MAGIC ```

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5')

# Test the tokenizer
chunk_example_text = "this is some text in a chunk"
encoded_input = tokenizer(chunk_example_text, padding=True, truncation=True, return_tensors='pt')
print(f"Number of tokens in `{chunk_example_text}`: {len(encoded_input['input_ids'][0])}")

# COMMAND ----------

# DBTITLE 1,Text Chunking UDF Writer
CHUNK_COLUMN_NAME = "chunked_text"
CHUNK_ID_COLUMN_NAME = "chunk_id"

# If using runtime < 14.3, remove `useArrow=True`
# useArrow=True which optimizes performance only works with 14.3+

# TODO: Add error handling
@F.udf(returnType=ArrayType(StringType())
          # useArrow=True, # set globally
          )
def split_char_recursive(content: str) -> List[str]:
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_size_tokens, chunk_overlap=chunk_overlap_tokens
    )
    chunks = text_splitter.split_text(content)
    return [doc for doc in chunks]


df_chunked = df_parsed.select(
    "*", F.explode(split_char_recursive("parsed_output.text")).alias(CHUNK_COLUMN_NAME)
).drop(F.col("parsed_output"))
df_chunked = df_chunked.select(
    "*", F.md5(F.col(CHUNK_COLUMN_NAME)).alias(CHUNK_ID_COLUMN_NAME)
)

df_chunked.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(gold_chunks_table_name)
display(df_chunked)

# Enable CDC for Vector Search Delta Sync
spark.sql(f"ALTER TABLE {gold_chunks_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Embed documents & sync to Vector Search index

# COMMAND ----------

# If index already exists, re-sync
try:
    w.vector_search_indexes.sync_index(index_name=gold_chunks_index_name)
# Otherwise, create new index
except ResourceDoesNotExist as ne_error:
    w.vector_search_indexes.create_index(
        name=gold_chunks_index_name,
        endpoint_name=vector_search_endpoint_name,
        primary_key=CHUNK_ID_COLUMN_NAME,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    embedding_model_endpoint_name=embedding_endpoint_name,
                    name=CHUNK_COLUMN_NAME,
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
            source_table=gold_chunks_table_name,
        ),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # View index status & output tables
# MAGIC
# MAGIC Your index is now embedding & syncing.  Time taken depends on the number of chunks.  You can view the status and how to query the index at the URL below.

# COMMAND ----------

# DBTITLE 1,Data Source URL Generator
def get_table_url(table_fqdn):
    split = table_fqdn.split(".")
    url = f"https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url

print("Vector index:\n")
print(w.vector_search_indexes.get_index(gold_chunks_index_name).status.message)
print("\nOutput tables:\n")
print(f"Bronze Delta Table w/ raw files: {get_table_url(bronze_raw_files_table_name)}")
print(f"Silver Delta Table w/ parsed files: {get_table_url(silver_parsed_files_table_name)}")
print(f"Gold Delta Table w/ chunked files: {get_table_url(gold_chunks_table_name)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Copy paste code for the RAG Chain YAML config
# MAGIC
# MAGIC * The following prints the configs used so that you can copy and paste them into your RAG YAML config

# COMMAND ----------

# DBTITLE 1,Vector Search RAG Configuration
rag_config = {
  "vector_search_endpoint_name": vector_search_endpoint_name,
  "vector_search_index": gold_chunks_index_name,
  "vector_search_schema": {
    "primary_key": CHUNK_ID_COLUMN_NAME,
    "chunk_text": CHUNK_COLUMN_NAME,
    "document_source": DOC_URI_COL_NAME
  },
  "vector_search_parameters": {
    "k": 3
  },
  "chunk_template": "`{chunk_text}`\n",
  "chat_endpoint": "databricks-dbrx-instruct",
  "chat_prompt_template": "You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
  "chat_prompt_template_variables": [
    "context",
    "question"
  ],
  "chat_endpoint_parameters": {
    "temperature": 0.01,
    "max_tokens": 500
  }
}

print("-----")
print("-----")
print("----- Copy this dict to `3_rag_chain_driver_notebook` ---")
print("-----")
print("-----")
print(rag_config)

# Convert the dictionary to a YAML string
yaml_str = yaml.dump(rag_config)

# Write the YAML string to a file
with open('2_rag_chain_config.yaml', 'w') as file:
    file.write(yaml_str)


# rag_config_yaml = f"""
# vector_search_endpoint_name: ""
# vector_search_index: "{gold_chunks_index_name}"
# # These must be set to use the Review App to match the columns in your index
# vector_search_schema:
#   primary_key: {CHUNK_ID_COLUMN_NAME}
#   chunk_text: {CHUNK_COLUMN_NAME}
#   document_source: {DOC_URI_COL_NAME}
# vector_search_parameters:
#   k: 3
# """

# print(rag_config_yaml)
