# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Document Prep Pipeline 
# MAGIC
# MAGIC <put intro>

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting Started
# MAGIC
# MAGIC 1. Update the configuration below.
# MAGIC 2. Press `Run All` to initialize the pipeline.
# MAGIC 3. Update the Notebook widgets to select the UC Catalog, Schema, and Volume.
# MAGIC 4. Press `Run All` (again) to execute the pipeline.
# MAGIC 5. Transfer the configuration output in the final cell to your RAG chain.

# COMMAND ----------

# MAGIC %md
# MAGIC # Initialize the pipeline
# MAGIC
# MAGIC Can take 5 - 10 minutes as this installs all required Python and system packages for all parsing/chunking strategies.  Hide the results of this cell.

# COMMAND ----------

# MAGIC %run ./initialize_pipeline

# COMMAND ----------

import json
import io
import yaml
import warnings
from abc import ABC, abstractmethod
from typing import List, TypedDict, Dict
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from databricks.sdk.service.serving import (EndpointStateReady)
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointStatusState,
    EndpointType,
    PipelineType,
    VectorIndexType,
)
from pyspark.sql import Column
from pyspark.sql.types import *
import pyspark.sql.functions as F
from mlflow.utils import databricks_utils as du

# Init workspace client
w = WorkspaceClient()

# Enable to test the strategies locally before applying in Spark
DEBUG = False

init_widgets()

# COMMAND ----------

# MAGIC %md
# MAGIC # START HERE: Configure your RAG Data Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC **See README.md for how to configure.**

# COMMAND ----------

# Embedding model is defined here b/c it is used in multiple places inside the `pipeline_configuration`
embedding_model = "BAAI/bge-large-en-v1.5"

pipeline_configuration = {
    # Short name of this configuration
    # Used as a postfix to identify the resulting Delta Tables e.g., `{uc_volume_name}_{tag}_gold`
    "tag": "bge_test_1",
    # Embedding model to use for embedding the chunks
    "embedding_model": {
        # model serving endpoint
        "endpoint": "databricks-bge-large-en",
        # name of the embedding model (maps to `embedding_model_configs`)
        "model_name": embedding_model,
    },
    # Parsing strategies that turn a raw document into a string
    # Each strategy must be a FileParser class defined in `parse_chunk_functions`
    "parsing_strategy": {
        "html": HTMLToMarkdownify(),
        "pdf": UnstructuredPDF(strategy="fast"),
        "pptx": UnstructuredPPTX(),
        "docx": PyPandocDocx(),
        "md": PassThroughNoParsing(),
    },
    # Chunking strategies that turned a parsed document into embeddable chunks
    # Each strategy must be a Chunker class defined in `parse_chunk_functions`
    # `default` will be used for any file extension with a defined strategy.
    "chunking_strategy": {
        "default": RecursiveTextSplitterByTokens(
            embedding_model_name=embedding_model,
            chunk_size_tokens=400,
            chunk_overlap_tokens=100,
        ),
        "md": MarkdownHeaderSplitter(),
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the configuration

# COMMAND ----------

load_configuration(pipeline_configuration)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widget-based configuration
# MAGIC
# MAGIC 1. Select a Vector Search endpoint
# MAGIC
# MAGIC If you do not have a Databricks Vector Search endpoint, follow these [steps](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) to create one.
# MAGIC
# MAGIC 2. Select UC Catalog, Schema, and UC Volume w/ your documents.
# MAGIC
# MAGIC Note: By default, the bronze/silver/gold Delta Tables with parsed chunks will land into this same UC Catalog/Schema.  You can change this behavior below.

# COMMAND ----------


vector_search_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")
uc_catalog_name = dbutils.widgets.get("uc_catalog_name")
uc_schema_name = dbutils.widgets.get("uc_schema_name")
source_uc_volume = f"/Volumes/{uc_catalog_name}/{uc_schema_name}/{dbutils.widgets.get('source_uc_volume')}"

validate_widget_values()

# COMMAND ----------

# MAGIC %md ## Output table & vector index names

# COMMAND ----------

# DBTITLE 1,Data Processing Workflow Manager
# Force this cell to re-run when these values are changed in the Notebook widgets
uc_catalog_name = dbutils.widgets.get("uc_catalog_name")
uc_schema_name = dbutils.widgets.get("uc_schema_name")
volume_raw_name = dbutils.widgets.get("source_uc_volume")

tag = pipeline_configuration['tag']

bronze_raw_files_table_name = (
    f"{uc_catalog_name}.{uc_schema_name}.{volume_raw_name}__{tag}__bronze_raw"
)
silver_parsed_files_table_name = (
    f"{uc_catalog_name}.{uc_schema_name}.{volume_raw_name}__{tag}__silver_parsed"
)
gold_chunks_table_name = (
    f"{uc_catalog_name}.{uc_schema_name}.{volume_raw_name}__{tag}__gold_chunked"
)
gold_chunks_index_name = (
    f"{uc_catalog_name}.{uc_schema_name}.{volume_raw_name}__{tag}__gold_chunked_index"
)

print(f"Bronze Delta Table w/ raw files: `{bronze_raw_files_table_name}`")
print(f"Silver Delta Table w/ parsed files: `{silver_parsed_files_table_name}`")
print(f"Gold Delta Table w/ chunked files: `{gold_chunks_table_name}`")
print(f"Vector Search Index mirror of Gold Delta Table: `{gold_chunks_index_name}`")

# COMMAND ----------

# MAGIC %md # Pipeline code

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze: Load the files from the UC Volume

# COMMAND ----------

# DBTITLE 1,Recursive PDF Ingestion Workflow
bronze_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .load(source_uc_volume)
)

# Rename the default column names to be more descriptive
bronze_df = (
    bronze_df.withColumnRenamed(LOADER_DEFAULT_DOC_URI_COL_NAME, DOC_URI_COL_NAME)
    .withColumnRenamed(LOADER_DEFAULT_BYTES_COL_NAME, BYTES_COL_NAME)
    .withColumnRenamed(LOADER_DEFAULT_BYTES_LENGTH_COL_NAME, BYTES_LENGTH_COL_NAME)
    .withColumnRenamed(LOADER_DEFAULT_MODIFICATION_TIME_COL_NAME, MODIFICATION_TIME_COL_NAME)
)

bronze_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    bronze_raw_files_table_name
)

# reload to get correct lineage in UC
bronze_df = spark.read.table(bronze_raw_files_table_name)

# display for debugging
display(bronze_df.drop(BYTES_COL_NAME))

if bronze_df.count() == 0:
    display(
        f"`{source_uc_volume}` does not contain any files.  Open the volume and upload at least file."
    )
    raise Exception(f"`{source_uc_volume}` does not contain any files.")

tag_delta_table(bronze_raw_files_table_name, pipeline_configuration)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: Parse the documents

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parser router
# MAGIC
# MAGIC This function decides, based on the `pipeline_configuration`, which parsers to run for each type of file in the UC Volume.

# COMMAND ----------

# The signature of the return type
parser_return_signature = StructType(
    [
        StructField(
            PARSED_OUTPUT_CONTENT_COL_NAME, StringType(), nullable=True
        ),  # Parsed content of the document
        StructField(
            PARSED_OUTPUT_STATUS_COL_NAME, StringType(), nullable=False
        ),  # SUCCESS if succeeded, `ERROR: {details}` otherwise
        StructField(
            PARSED_OUTPUT_METADATA_COL_NAME, StringType(), nullable=False
        ),  # The parser that was used
    ]
)


# Router function to select parsing strategy based on the config
def parse_file_wrapper(doc_uri, raw_doc_string_content, user_config):
    file_extension = doc_uri.split(".")[-1]

    # check if file extension can be extracted from the doc_uri
    if file_extension is None or file_extension == "":
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: None,
            PARSED_OUTPUT_STATUS_COL_NAME: f"ERROR: Could not determine file extension of file `{doc_uri}`",
            PARSED_OUTPUT_METADATA_COL_NAME: "None",
        }

    # check if the config specifies a parser for this file_extension
    parser_class = user_config["parsing_strategy"].get(file_extension)
    if parser_class is None:
        return {
            PARSED_OUTPUT_CONTENT_COL_NAME: None,
            PARSED_OUTPUT_STATUS_COL_NAME: f"ERROR: No parsing strategy for file extension `{file_extension}`",
            PARSED_OUTPUT_METADATA_COL_NAME: "None",
        }

    try:
        parsed_output = parser_class.parse_bytes(raw_doc_string_content)
        parsed_output[PARSED_OUTPUT_METADATA_COL_NAME] = str(parser_class)
        return parsed_output
    except Exception as e:
        return {
            "doc_parsed_content": None,
            "status": f"ERROR: {e}",
            PARSED_OUTPUT_METADATA_COL_NAME: "None"
        }


# Create the UDF, directly passing the user's provided configuration stored in `pipeline_configuration`
parse_file_udf = udf(
    lambda doc_uri, raw_doc_string_content: parse_file_wrapper(
        doc_uri, raw_doc_string_content, pipeline_configuration
    ),
    parser_return_signature,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Debug the individual parsers
# MAGIC
# MAGIC Use this function if you want to test new parsers locally without Spark *before* you run them parrelized in Spark.

# COMMAND ----------

if DEBUG:
  test_sample = bronze_df.limit(1).collect()

  for sample in test_sample:
    test_output = parse_file_wrapper(test_sample[0][DOC_URI_COL_NAME], test_sample[0][BYTES_COL_NAME], pipeline_configuration)
    print(test_output)
    print(test_output.keys())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the parsers in Spark
# MAGIC
# MAGIC This cell runs the configured parsers in parallel via Spark.  Inspect the outputs to verify that parsing is working correctly.

# COMMAND ----------

# Run the parsing
df_parsed = bronze_df.withColumn(
    PARSED_OUTPUT_STRUCT_COL_NAME,
    parse_file_udf(F.col(DOC_URI_COL_NAME), F.col(BYTES_COL_NAME)),
)

# TODO: Temporarily cache ^^ to speed up the pipeline so it doesn't recompute on every computation.

# Check and warn on any errors
errors_df = df_parsed.filter(
    F.col(f"{PARSED_OUTPUT_STRUCT_COL_NAME}.{PARSED_OUTPUT_STATUS_COL_NAME}")
    != "SUCCESS"
)
num_errors = errors_df.count()
if num_errors > 0:
    print(f"{num_errors} documents had parse errors.  Please review.")
    display(errors_df)

# Move the parsed contents into a non-struct column, dropping the status
df_parsed = (
    df_parsed.filter(
        F.col(f"{PARSED_OUTPUT_STRUCT_COL_NAME}.{PARSED_OUTPUT_STATUS_COL_NAME}")
        == "SUCCESS"
    )
    .withColumn(
        PARSED_OUTPUT_CONTENT_COL_NAME,
        F.col(f"{PARSED_OUTPUT_STRUCT_COL_NAME}.{PARSED_OUTPUT_CONTENT_COL_NAME}"),
    )
    .drop(PARSED_OUTPUT_STRUCT_COL_NAME)
    .drop(BYTES_COL_NAME)
)

df_parsed.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    silver_parsed_files_table_name
)

# reload to get correct lineage in UC and to filter out any error rows for the downstream step.
df_parsed = spark.read.table(silver_parsed_files_table_name)

print(f"Parsed {df_parsed.count()} documents.")

display(df_parsed)

tag_delta_table(silver_parsed_files_table_name, pipeline_configuration)

# COMMAND ----------

# MAGIC %md ## Gold: Chunk the parsed text

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunker router
# MAGIC
# MAGIC This function decides, based on the `pipeline_configuration`, which chunkers to run for each type of file in the UC Volume.

# COMMAND ----------

# The signature of the return type
chunker_return_signature = StructType(
    [
        StructField(
            CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME,
            ArrayType(StringType()),
            nullable=True,
        ),  # Parsed content of the document
        StructField(
            CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME, StringType(), nullable=False
        ),  # SUCCESS if succeeded, `ERROR: {details}` otherwise
        StructField(
            CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME, StringType(), nullable=False
        ),  # The chunker that was used
    ]
)


# Router function to select parsing strategy based on the config
def chunker_wrapper(doc_uri, doc_parsed_contents, user_config):
    file_extension = doc_uri.split(".")[-1]

    # check if file extension can be extracted from the doc_uri
    if file_extension is None or file_extension == "":
        return {
            CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME: [],
            CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME: f"ERROR: Could not determine file extension of file `{doc_uri}`",
            CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME: "None",
        }

    # Use file_extension's configuration or the default
    chunker_class = user_config["chunking_strategy"].get(file_extension) or user_config[
        "chunking_strategy"
    ].get("default")
    if chunker_class is None:
        return {
            CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME: [],
            CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME: f"ERROR: No chunking strategy for file extension `{file_extension}`; no default strategy provided.",
            CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME: "None",
        }

    try:
        output_chunks = chunker_class.chunk_parsed_content(doc_parsed_contents)
        output_chunks[CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME] = str(chunker_class)
        return output_chunks
    except Exception as e:
        return {
            CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME: None,
            CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME: f"ERROR: {e}",
            CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME: "None",
        }


# Create the UDF, directly passing the user's provided configuration stored in `pipeline_configuration`
chunk_file_udf = F.udf(
    lambda doc_uri, doc_parsed_contents: chunker_wrapper(
        doc_uri, doc_parsed_contents, pipeline_configuration
    ),
    chunker_return_signature,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Debug the individual chunkers
# MAGIC
# MAGIC Use this function if you want to test new chunkers locally without Spark *before* you run them parrelized in Spark.

# COMMAND ----------

if DEBUG:
  test_sample = df_parsed.limit(1).collect()

  for sample in test_sample:
    test_output = chunker_wrapper(test_sample[0][DOC_URI_COL_NAME], test_sample[0][PARSED_OUTPUT_CONTENT_COL_NAME], pipeline_configuration)
    print(test_output)
    print(test_output.keys())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the chunkers in Spark
# MAGIC
# MAGIC This cell runs the configured chunkers in parallel via Spark.  Inspect the outputs to verify that parsing is working correctly.

# COMMAND ----------

# DBTITLE 1,Text Chunking UDF Writer
df_chunked = df_parsed.withColumn(
    CHUNKED_OUTPUT_STRUCT_COL_NAME,
    chunk_file_udf(F.col(DOC_URI_COL_NAME), F.col(PARSED_OUTPUT_CONTENT_COL_NAME)),
)

# Check and warn on any errors
errors_df = df_chunked.filter(
    F.col(f"{CHUNKED_OUTPUT_STRUCT_COL_NAME}.{CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME}")
    != "SUCCESS"
)
num_errors = errors_df.count()
if num_errors > 0:
    print(f"{num_errors} chunks had parse errors.  Please review.")
    display(errors_df)

df_chunked = df_chunked.filter(
    F.col(f"{CHUNKED_OUTPUT_STRUCT_COL_NAME}.{CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME}")
    == "SUCCESS"
)

# Flatten the chunk arrays and rename columns
df_chunked = (
    df_chunked.withColumn(
        CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME,
        F.col(
            f"{CHUNKED_OUTPUT_STRUCT_COL_NAME}.{CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME}"
        ),
    )
    .withColumn(
        CHUNK_TEXT_COL_NAME,
        F.explode(
            F.col(
                f"{CHUNKED_OUTPUT_STRUCT_COL_NAME}.{CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME}"
            )
        ),
    )
    .withColumnRenamed(PARSED_OUTPUT_CONTENT_COL_NAME, FULL_DOC_PARSED_OUTPUT_COL_NAME)
).drop(F.col(CHUNKED_OUTPUT_STRUCT_COL_NAME))


# Add a unique ID for each chunk
df_chunked = df_chunked.withColumn(CHUNK_ID_COL_NAME, F.md5(F.col(CHUNK_TEXT_COL_NAME)))

# Write to Delta Table
df_chunked.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    gold_chunks_table_name
)

# Enable CDC for Vector Search Delta Sync
spark.sql(
    f"ALTER TABLE {gold_chunks_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

print(f"Produced a total of {df_chunked.count()} chunks.")

# Display without the parent document text - this is saved to the Delta Table
display(df_chunked.drop(FULL_DOC_PARSED_OUTPUT_COL_NAME))

tag_delta_table(gold_chunks_table_name, pipeline_configuration)


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
        primary_key=CHUNK_ID_COL_NAME,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    embedding_model_endpoint_name=pipeline_configuration['embedding_model']['endpoint'],
                    name=CHUNK_TEXT_COL_NAME,
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
            source_table=gold_chunks_table_name,
        ),
    )

tag_delta_table(gold_chunks_index_name, pipeline_configuration)

# COMMAND ----------

# MAGIC %md
# MAGIC ### View index status & output tables
# MAGIC
# MAGIC Your index is now embedding & syncing.  Time taken depends on the number of chunks.  You can view the status and how to query the index at the URL below.

# COMMAND ----------

# DBTITLE 1,Data Source URL Generator
def get_table_url(table_fqdn):
    split = table_fqdn.split(".")
    browser_url = du.get_browser_hostname()
    url = f"{browser_url}/explore/data/{split[0]}/{split[1]}/{split[2]}"
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
    "primary_key": CHUNK_ID_COL_NAME,
    "chunk_text": CHUNK_TEXT_COL_NAME,
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
  },
  "data_pipeline_config": stringify_config(pipeline_configuration)
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
with open('rag_chain_config.yaml', 'w') as file:
    file.write(yaml_str)
