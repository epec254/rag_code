# Databricks notebook source
# MAGIC %md
# MAGIC # Unpack inference table to request_log & assessment_log
# MAGIC
# MAGIC IMPORTANT NOTE: This unpacking now happens automatically every ~1 hour.  If you ned to see logs from your Review App or Endpoint faster than 1 hour, you can use this Notebook.
# MAGIC
# MAGIC Use this notebook to turn your chain's Inference Table into a well-formed `request_log` and `assessment_log` table.
# MAGIC
# MAGIC **Important: Inference Tables can take 15 - 30 minutes to reflect requests made to the REST API / Review App.**
# MAGIC
# MAGIC Known issues:
# MAGIC - Every run of this notebook overwrites the previous tables.

# COMMAND ----------

# DBTITLE 1,PySpark DataFrame Utilities
from pyspark.sql import DataFrame, SparkSession, window, functions as F, types as T
from typing import Optional, Tuple

# COMMAND ----------

# DBTITLE 1,Configure table names
uc_catalog = 'catalog'
uc_schema = 'schema'
model_name = 'pdf_bot'

############
# UC FQN to the Inference Table
# You can find this from the chain's Model Serving Endpoint, but by default it is set to `rag_studio-{model_name}_payload`
############
inference_table_uc_fqn = f"{uc_catalog}.{uc_schema}.`{model_name}_payload`"

############
# Specify UC FQN to output the `request_log` table to
############
request_log_output_uc_fqn = f"{uc_catalog}.{uc_schema}.`{model_name}_request_log`"


############
# Specify UC FQN to output the `assessment_log` table to
############
assessment_log_output_uc_fqn = f"{uc_catalog}.{uc_schema}.`{model_name}_assessment_log`"

# COMMAND ----------

from pyspark.sql import types as T


######################################################################
# Request log schema definitions
######################################################################

# Format of the conversation following the OpenAI messages format.
MESSAGE_SCHEMA = T.StructType(
    [
        T.StructField("role", T.StringType()),
        T.StructField("content", T.StringType()),
    ]
)

# Format of the RAG response in the choices format.
CHOICES_SCHEMA = T.ArrayType(T.StructType([T.StructField("message", MESSAGE_SCHEMA)]))

# Schema of a single retrieval chunk in the trace step.
CHUNK_SCHEMA = T.StructType(
    [
        T.StructField("chunk_id", T.StringType()),
        T.StructField("doc_uri", T.StringType()),
        T.StructField("content", T.StringType()),
    ]
)

RETRIEVAL_SCHEMA = T.StructType(
    [
        T.StructField("query_text", T.StringType()),
        T.StructField("chunks", T.ArrayType(CHUNK_SCHEMA)),
    ]
)

TEXT_GENERATION_SCHEMA = T.StructType(
    [
        T.StructField("prompt", T.StringType()),
        T.StructField("generated_text", T.StringType()),
    ]
)

# Schema for an individual trace step.
TRACE_STEP_SCHEMA = T.StructType(
    [
        T.StructField("step_id", T.StringType()),
        T.StructField("name", T.StringType()),
        T.StructField("type", T.StringType()),
        T.StructField("start_timestamp", T.TimestampType()),
        T.StructField("end_timestamp", T.TimestampType()),
        T.StructField("retrieval", RETRIEVAL_SCHEMA),
        T.StructField("text_generation", TEXT_GENERATION_SCHEMA),
    ]
)

# Schema of the "trace" field in the final request logs table.
TRACE_SCHEMA = T.StructType(
    [
        T.StructField("app_version_id", T.StringType()),
        T.StructField("start_timestamp", T.TimestampType()),
        T.StructField("end_timestamp", T.TimestampType()),
        T.StructField("is_truncated", T.BooleanType()),
        T.StructField("steps", T.ArrayType(TRACE_STEP_SCHEMA)),
    ]
)

MESSAGES_SCHEMA = T.ArrayType(MESSAGE_SCHEMA)

REQUEST_SCHEMA = T.StructType(
    [
        T.StructField("request_id", T.StringType()),
        T.StructField("conversation_id", T.StringType()),
        T.StructField("timestamp", T.TimestampType()),
        T.StructField("messages", MESSAGES_SCHEMA),
        T.StructField("last_input", T.StringType()),
    ]
)

# Full schema of the final request logs table.
REQUEST_LOG_SCHEMA = T.StructType(
    [
        T.StructField("request", REQUEST_SCHEMA),
        T.StructField("trace", TRACE_SCHEMA),
        T.StructField(
            "output",
            T.StructType([T.StructField("choices", CHOICES_SCHEMA)]),
        ),
    ]
)


######################################################################
# Assessment log schema definitions
######################################################################

##################### Common schema definitions ######################
# Schema of the source tags in the assessment log.
ASSESSMENT_SOURCE_TAGS_SCHEMA = T.MapType(T.StringType(), T.StringType())

# Schema of the source field in the assessment log.
ASSESSMENT_SOURCE_SCHEMA = T.StructType(
    [
        T.StructField("type", T.StringType()),
        T.StructField("id", T.StringType()),
        T.StructField("tags", ASSESSMENT_SOURCE_TAGS_SCHEMA),
    ]
)

##################### Proto schema definitions #######################
# Schema of the assessments w.r.t. the proto definitions, which vary slightly from
# the final table schemas (particularly around the value of the ratings map).
RATING_VALUE_PROTO_SCHEMA = T.StructType(
    [
        T.StructField("value", T.BooleanType()),
        T.StructField("rationale", T.StringType()),
    ]
)

COMMON_ASSESSMENT_PROTO_SCHEMA = T.StructType(
    [
        T.StructField("step_id", T.StringType()),
        T.StructField("ratings", T.MapType(T.StringType(), RATING_VALUE_PROTO_SCHEMA)),
        T.StructField("free_text_comment", T.StringType()),
    ]
)

TEXT_ASSESSMENT_PROTO_SCHEMA = T.StructType(
    [
        *COMMON_ASSESSMENT_PROTO_SCHEMA,
        T.StructField("suggested_output", T.StringType()),
    ]
)

RETRIEVAL_ASSESSMENT_PROTO_SCHEMA = T.StructType(
    [
        T.StructField("position", T.IntegerType()),
        *COMMON_ASSESSMENT_PROTO_SCHEMA,
    ]
)

# Schema of the assessment protos from the inference table's dataframe_records.
ASSESSMENT_PROTO_SCHEMA = T.ArrayType(
    T.StructType(
        [
            T.StructField("request_id", T.StringType()),
            T.StructField("step_id", T.StringType()),
            T.StructField("source", ASSESSMENT_SOURCE_SCHEMA),
            T.StructField(
                "text_assessments", T.ArrayType(TEXT_ASSESSMENT_PROTO_SCHEMA)
            ),
            T.StructField(
                "retrieval_assessments", T.ArrayType(RETRIEVAL_ASSESSMENT_PROTO_SCHEMA)
            ),
        ]
    )
)


##################### Table schema definitions #######################
RATING_VALUE_TABLE_SCHEMA = T.StructType(
    [
        T.StructField("bool_value", T.BooleanType()),
        T.StructField("double_value", T.DoubleType()),
        T.StructField("rationale", T.StringType()),
    ]
)

# Fields of the assessment structs that are common to both text and retrieval assessments.
COMMON_ASSESSMENT_TABLE_SCHEMA = [
    T.StructField("step_id", T.StringType()),
    T.StructField(
        "ratings",
        T.MapType(
            T.StringType(),
            RATING_VALUE_TABLE_SCHEMA,
        ),
    ),
    T.StructField("free_text_comment", T.StringType()),
]

# Schema of text assessments.
TEXT_ASSESSMENT_TABLE_SCHEMA = T.StructType(
    [
        *COMMON_ASSESSMENT_TABLE_SCHEMA,
        T.StructField("suggested_output", T.StringType()),
    ]
)

# Schema of retrieval assessments.
RETRIEVAL_ASSESSMENT_TABLE_SCHEMA = T.StructType(
    [
        T.StructField("position", T.IntegerType()),
        *COMMON_ASSESSMENT_TABLE_SCHEMA,
    ]
)

# Full schema of the final assessment logs table.
ASSESSMENT_LOG_SCHEMA = T.StructType(
    [
        T.StructField("request_id", T.StringType()),
        T.StructField("step_id", T.StringType()),
        T.StructField("source", ASSESSMENT_SOURCE_SCHEMA),
        T.StructField("timestamp", T.TimestampType()),
        T.StructField("text_assessment", TEXT_ASSESSMENT_TABLE_SCHEMA),
        T.StructField("retrieval_assessment", RETRIEVAL_ASSESSMENT_TABLE_SCHEMA),
    ]
)



# COMMAND ----------

def unpack_and_split_payloads(payload_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Unpacks the request and assessment payloads from the given DataFrame
    and splits them into separate request log and assessment log DataFrames.

    :param payload_df: A DataFrame containing payloads to unpack and split
    :return: A tuple containing (request logs DataFrame, assessment logs DataFrame)
    """
    payloads = payload_df.filter(
        F.col("status_code") == "200"
    ).withColumn(  # Ignore error requests
        "timestamp", (F.col("timestamp_ms") / 1000).cast("timestamp")
    )

    # Split the payloads into requests and assessments based on the payload structure
    request_payloads = payloads.filter(F.expr("response:choices IS NOT NULL"))
    assessment_payloads = payloads.filter(F.expr("response:choices IS NULL"))

    # Unpack the requests
    request_logs = (
        request_payloads.withColumn(
            "request",
            F.struct(
                F.col("databricks_request_id").alias("request_id"),
                F.expr("request:databricks_options.conversation_id").alias(
                    "conversation_id"
                ),
                F.col("timestamp"),
                F.from_json(F.expr("request:messages"), MESSAGES_SCHEMA).alias(
                    "messages"
                ),
                F.element_at(
                    F.from_json(F.expr("request:messages"), MESSAGES_SCHEMA), -1
                )
                .getItem("content")
                .alias("last_input"),
            ),
        )
        .withColumn(
            "trace",
            F.from_json(F.expr("response:databricks_output.trace"), TRACE_SCHEMA),
        )
        .withColumn(
            "output",
            F.struct(
                F.from_json(F.expr("response:choices"), CHOICES_SCHEMA).alias("choices")
            ),
        )
        .select("request", "trace", "output")
    )

    # Unpack the assessments
    assessment_logs = (
        assessment_payloads.withColumn(
            "assessments",
            F.explode(
                F.from_json(
                    F.expr("request:dataframe_records"), ASSESSMENT_PROTO_SCHEMA
                )
            ),
        )
        .withColumn(
            "text_assessments",
            # Transform the list of text assessments into a list of assessment structs (with empty
            # retrieval assessments) so we can concatenate them before exploding.
            # The ordering of the structs must match exactly to concatenate them.
            F.transform(
                F.col("assessments.text_assessments"),
                lambda ta: F.struct(
                    # Transform the proto ratings map (which only has a boolean value)
                    # to the table ratings map (which has bool_value and double_value).
                    F.struct(
                        ta.step_id,
                        F.transform_values(
                            ta.ratings,
                            lambda _, rating_val: F.struct(
                                rating_val.value.alias("bool_value"),
                                F.lit(None).cast(T.DoubleType()).alias("double_value"),
                                rating_val.rationale,
                            ),
                        ).alias("ratings"),
                        ta.free_text_comment,
                        ta.suggested_output,
                    ).alias("text_assessment"),
                    F.lit(None)
                    .cast(RETRIEVAL_ASSESSMENT_TABLE_SCHEMA)
                    .alias("retrieval_assessment"),
                ),
            ),
        )
        .withColumn(
            "retrieval_assessments",
            # Transform the list of retrieval assessments into a list of assessment structs (with empty
            # text assessments) so we can concatenate them before exploding.
            # The ordering of the structs must match exactly to concatenate them.
            F.transform(
                F.col("assessments.retrieval_assessments"),
                lambda ra: F.struct(
                    F.lit(None)
                    .cast(TEXT_ASSESSMENT_TABLE_SCHEMA)
                    .alias("text_assessment"),
                    # Transform the proto ratings map (which only has a boolean value)
                    # to the table ratings map (which has bool_value and double_value).
                    F.struct(
                        ra.position,
                        ra.step_id,
                        F.transform_values(
                            ra.ratings,
                            lambda _, rating_val: F.struct(
                                rating_val.value.alias("bool_value"),
                                F.lit(None).cast(T.DoubleType()).alias("double_value"),
                                rating_val.rationale,
                            ),
                        ).alias("ratings"),
                        ra.free_text_comment,
                    ).alias("retrieval_assessment"),
                ),
            ),
        )
        .withColumn(
            "all_assessments",
            F.explode(
                F.concat(
                    # Coalesce with an empty array to handle cases where only one of
                    # text_assessments or retrieval_assessments were passed.
                    F.coalesce(F.col("text_assessments"), F.array()),
                    F.coalesce(F.col("retrieval_assessments"), F.array()),
                )
            ),
        )
        .select(
            "assessments.request_id",
            F.coalesce(
                F.col("all_assessments.text_assessment.step_id"),
                F.col("all_assessments.retrieval_assessment.step_id"),
            ).alias("step_id"),
            "assessments.source",
            "timestamp",
            "all_assessments.text_assessment",
            "all_assessments.retrieval_assessment",
        )
    )
    return request_logs, assessment_logs

def dedup_assessment_logs(
    assessment_logs: DataFrame, granularity: Optional[str] = None
) -> DataFrame:
    """
    Deduplicates the given assessment logs DataFrame by only keeping the latest entry matching
    a given request_id and step_id.

    :param assessment_logs: Assessment logs to deduplicate
    :param granularity: Time granularity of the deduplication (e.g., remove all duplicates per "hour").
                        String value here should conform to a format string of pyspark.sql.functions.date_trunc
                        (https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.functions.date_trunc.html).
                        Currently, only "hour" is supported.
                        If None, deduplication is done across the entire dataset.
    :return: Filtered DataFrame of assessment logs
    """
    _ROW_NUM_COL = "row_num"
    _TRUNCATED_TIME_COL = "truncated_time"
    _ASSESSMENT_LOG_PRIMARY_KEYS = [
        F.col("request_id"),
        F.col("step_id"),
        F.col("source.id"),
        # Retrieval assessments are additionally identified by their chunk position.
        F.coalesce(F.col("retrieval_assessment.position"), F.lit(None)),
    ]
    _SUPPORTED_GRANULARITIES = ["hour"]

    if granularity is not None and granularity not in _SUPPORTED_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {_SUPPORTED_GRANULARITIES} or None, but got {granularity}"
        )

    partition_cols = _ASSESSMENT_LOG_PRIMARY_KEYS + (
        [F.date_trunc(granularity, "timestamp")] if granularity is not None else []
    )
    window_spec = window.Window.partitionBy(partition_cols).orderBy(F.desc("timestamp"))

    # Use row_number() to assign a rank to each row within the window
    assessments_ranked = assessment_logs.withColumn(
        _ROW_NUM_COL, F.row_number().over(window_spec)
    )

    # Filter the rows where row_num is 1 to keep only the latest timestamp
    return assessments_ranked.filter(F.col(_ROW_NUM_COL) == 1).drop(
        _ROW_NUM_COL, _TRUNCATED_TIME_COL
    )

# COMMAND ----------

# DBTITLE 1,Unpack the payloads
# Unpack the payloads
payload_df = spark.table(inference_table_uc_fqn)
request_logs, raw_assessment_logs = unpack_and_split_payloads(payload_df)

# The Review App logs every user interaction with the feedback widgets to the inference table - this code de-duplicates them
deduped_assessments_df = dedup_assessment_logs(raw_assessment_logs, granularity="hour")
deduped_assessments_df.write.format("delta").mode("overwrite").saveAsTable(assessment_log_output_uc_fqn)

# COMMAND ----------

display(request_logs)

# COMMAND ----------

display(deduped_assessments_df)

# COMMAND ----------

# DBTITLE 1,Write raw logs
request_logs.write.format("delta").option("mergeSchema", "true").mode(
    "overwrite"
).saveAsTable(request_log_output_uc_fqn)
deduped_assessments_df.write.format("delta").option("mergeSchema", "true").mode(
    "overwrite"
).saveAsTable(assessment_log_output_uc_fqn)

# COMMAND ----------

def get_table_url(table_fqdn):
    split = table_fqdn.split(".")
    url = f"https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url


print(f"Wrote `request_log` to: {get_table_url(request_log_output_uc_fqn).replace('`', '')}")
print(f"Wrote `assessment_log` to: {get_table_url(assessment_log_output_uc_fqn).replace('`', '')}")
