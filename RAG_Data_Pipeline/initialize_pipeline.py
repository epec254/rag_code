# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk mlflow

# COMMAND ----------

# DBTITLE 1,Column name constants
# Bronze table
DOC_URI_COL_NAME = "doc_uri"
CONTENT_COL_NAME = "raw_doc_contents_string"
BYTES_COL_NAME = "raw_doc_contents_bytes"
BYTES_LENGTH_COL_NAME = "raw_doc_bytes_length"
MODIFICATION_TIME_COL_NAME = "raw_doc_modification_time"

# Bronze table auto loader names
LOADER_DEFAULT_DOC_URI_COL_NAME = "path"
LOADER_DEFAULT_BYTES_COL_NAME = "content"
LOADER_DEFAULT_BYTES_LENGTH_COL_NAME = "length"
LOADER_DEFAULT_MODIFICATION_TIME_COL_NAME = "modificationTime"

# Silver table
PARSED_OUTPUT_STRUCT_COL_NAME = "parser_output"
PARSED_OUTPUT_CONTENT_COL_NAME = "doc_parsed_contents"
PARSED_OUTPUT_STATUS_COL_NAME = "parser_status"
PARSED_OUTPUT_METADATA_COL_NAME = "parser_metadata"

# Gold table

# intermediate values
CHUNKED_OUTPUT_STRUCT_COL_NAME = "chunker_output"
CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME = "chunked_texts"
CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME = "chunker_status"
CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME = "chunker_metadata"

FULL_DOC_PARSED_OUTPUT_COL_NAME = "parent_doc_parsed_contents"
CHUNK_TEXT_COL_NAME = "chunk_text"
CHUNK_ID_COL_NAME = "chunk_id"

# COMMAND ----------

# DBTITLE 1,Load parsing funcs
# MAGIC %run ./parse_chunk_functions

# COMMAND ----------

# DBTITLE 1,Install librariesa
# Install PIP packages & APT-GET libraries for all parsers/chunkers.
# This can take a while on smaller clusters.  If you plan to only use a subset of the parsing/chunking strategies, you can optimize this by only installing the packages for those parsers/chunkers.
install_pip_and_aptget_packages_for_all_parsers_and_chunkers()

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load parsing funcs
# MAGIC %run ./parse_chunk_functions

# COMMAND ----------

# DBTITLE 1,Column name constants
# Reload constants after notebook restarts

# Bronze table
DOC_URI_COL_NAME = "doc_uri"
CONTENT_COL_NAME = "raw_doc_contents_string"
BYTES_COL_NAME = "raw_doc_contents_bytes"
BYTES_LENGTH_COL_NAME = "raw_doc_bytes_length"
MODIFICATION_TIME_COL_NAME = "raw_doc_modification_time"

# Bronze table auto loader names
LOADER_DEFAULT_DOC_URI_COL_NAME = "path"
LOADER_DEFAULT_BYTES_COL_NAME = "content"
LOADER_DEFAULT_BYTES_LENGTH_COL_NAME = "length"
LOADER_DEFAULT_MODIFICATION_TIME_COL_NAME = "modificationTime"

# Silver table
PARSED_OUTPUT_STRUCT_COL_NAME = "parser_output"
PARSED_OUTPUT_CONTENT_COL_NAME = "doc_parsed_contents"
PARSED_OUTPUT_STATUS_COL_NAME = "parser_status"
PARSED_OUTPUT_METADATA_COL_NAME = "parser_metadata"

# Gold table

# intermediate values
CHUNKED_OUTPUT_STRUCT_COL_NAME = "chunker_output"
CHUNKED_OUTPUT_ARRAY_OF_CHUNK_TEXT_COL_NAME = "chunked_texts"
CHUNKED_OUTPUT_CHUNKER_STATUS_COL_NAME = "chunker_status"
CHUNKED_OUTPUT_CHUNKER_METADATA_COL_NAME = "chunker_metadata"

FULL_DOC_PARSED_OUTPUT_COL_NAME = "parent_doc_parsed_contents"
CHUNK_TEXT_COL_NAME = "chunk_text"
CHUNK_ID_COL_NAME = "chunk_id"

# COMMAND ----------

# DBTITLE 1,PIP imports
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

# Use optimizations if available
dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
if dbr_majorversion >= 14:
  spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

# COMMAND ----------

# DBTITLE 1,Config helpers - stringify
# Configuration represented as strings
def stringify_config(config):
    stringed_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively call the function for nested dictionaries
            stringed_config[key] = stringify_config(value)
        else:
            # Convert the value to string
            stringed_config[key] = str(value)
    return stringed_config


def tag_delta_table(table_fqn, config):
    sqls = [f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("rag_data_pipeline_tag" = "{config['tag']}")
        """, f"""
        ALTER TABLE {table_fqn}
        SET TAGS ("table_source" = "rag_data_pipeline")
        """]
    for sql in sqls:
        spark.sql(sql)




# COMMAND ----------

# DBTITLE 1,Config helpers - validation
def validate_config(pipeline_configuration):
  # Check for correct keys in the config
  allowed_config_keys = set(
      ["tag", "embedding_model", "parsing_strategy", "chunking_strategy"]
  )
  config_keys = set(pipeline_configuration.keys())
  extra_keys = config_keys - allowed_config_keys
  missing_keys = allowed_config_keys - config_keys

  if len(missing_keys) > 0:
      raise ValueError(
          f"PROBLEM: `pipeline_configuration` has missing keys.  \n SOLUTION: Add the missing keys {missing_keys}."
      )

  if len(extra_keys) > 0:
      raise ValueError(
          f"PROBLEM: `pipeline_configuration` has extra keys.  \n SOLUTION: Remove the extra keys {extra_keys}."
      )


  # Check embedding model
  if (
      pipeline_configuration["embedding_model"]["model_name"]
      not in EMBEDDING_MODELS.keys()
  ):
      raise ValueError(
          f"PROBLEM: Embedding model {pipeline_configuration['embedding_model']['model_name']} not configured.\nSOLUTION: Update `EMBEDDING_MODELS` in the `parse_chunk_functions` notebook."
      )

  # Check embedding model endpoint
  # TODO: Validate the endpoint is a valid embeddings endpoint
  try:
      endpoint = w.serving_endpoints.get(
          pipeline_configuration["embedding_model"]["endpoint"]
      )
      if endpoint.state.ready != EndpointStateReady.READY:
          browser_url = du.get_browser_hostname()
          raise ValueError(
              f"PROBLEM: Embedding model serving endpoint `{pipeline_configuration['embedding_model']['endpoint']}` exists, but is not ready.  SOLUTION: Visit the endpoint's page at https://{browser_url}/ml/endpoints/{pipeline_configuration['embedding_model']['endpoint']} to debug why it is not ready."
          )
  except ResourceDoesNotExist as e:
      raise ValueError(
          f"PROBLEM: Embedding model serving endpoint `{pipeline_configuration['embedding_model']['endpoint']}` does not exist.  SOLUTION: Either [1] Check that the name of the endpoint is valid.  [2] Deploy the embedding model using the `create_embedding_endpoint` notebook."
      )

# COMMAND ----------

# DBTITLE 1,Config helpers - Load configs
def load_configuration(pipeline_configuration):
  for item, strategy in pipeline_configuration['chunking_strategy'].items():
    print(f"Loading {strategy}...")
    if not strategy.load():
      raise Exception(f"Failed to load {strategy}...")

  for item, strategy in pipeline_configuration['parsing_strategy'].items():
    print(f"Loading up {strategy}...")
    if not strategy.load():
      raise Exception(f"Failed to load {strategy}...")

# COMMAND ----------

# DBTITLE 1,Init user input widgets
def init_vs_widgets():
    vector_search_endpoints_in_workspace = [
        item.name
        for item in w.vector_search_endpoints.list_endpoints()
        if item.endpoint_status.state == EndpointStatusState.ONLINE
    ]

    if len(vector_search_endpoints_in_workspace) == 0:
        raise Exception(
            "No Vector Search Endpoints are online in this workspace.  Please follow the instructions here to create a Vector Search endpoint: https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint"
        )

    # Vector Search Endpoint Widget
    if (
        len(vector_search_endpoints_in_workspace) > 1024
    ):  # use text widget if number of values > 1024
        dbutils.widgets.text(
            "vector_search_endpoint_name",
            defaultValue="",
            label="#1 VS endpoint",
        )
    else:
        dbutils.widgets.dropdown(
            "vector_search_endpoint_name",
            defaultValue="",
            choices=vector_search_endpoints_in_workspace + [""],
            label="#1 Select VS endpoint",
        )


def init_uc_widgets():
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
        uc_schemas = [
            schema for schema in uc_schemas if schema != "__databricks_internal"
        ]

        if len(uc_schemas) > 1024:  # use text widget if number of values > 1024
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


def init_widgets():
    init_uc_widgets()
    init_vs_widgets()

# COMMAND ----------

# DBTITLE 1,Validate user input widgets
def validate_widget_values():
    # Vector Search
    vector_search_endpoint_name = dbutils.widgets.get("vector_search_endpoint_name")
    if vector_search_endpoint_name == "" or vector_search_endpoint_name is None:
        raise Exception("Please select a Vector Search endpoint to continue.")
    else:
        print(f"Using `{vector_search_endpoint_name}` as the Vector Search endpoint.")

    # UC
    uc_catalog_name = dbutils.widgets.get("uc_catalog_name")
    uc_schema_name = dbutils.widgets.get("uc_schema_name")
    source_uc_volume = f"/Volumes/{uc_catalog_name}/{uc_schema_name}/{dbutils.widgets.get('source_uc_volume')}"

    
    if (uc_catalog_name == "" or uc_catalog_name is None) or (
        uc_schema_name == "" or uc_schema_name is None
    ):
        raise Exception("Please select a UC Catalog & Schema to continue.")
    else:
        print(f"Using `{uc_catalog_name}.{uc_schema_name}` as the UC Catalog / Schema.")

    if source_uc_volume == "" or source_uc_volume is None:
        raise Exception("Please select a source UC Volume w/ documents to continue.")
    else:
        print(f"Using {source_uc_volume} as the UC Volume Source.")
