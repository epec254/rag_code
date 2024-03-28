# Databricks notebook source
# MAGIC %md
# MAGIC # Unpack inference table to request_log & assessment_log
# MAGIC
# MAGIC ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
# MAGIC
# MAGIC Use this notebook to turn your chain's Inference Table into a well-formed `request_log` and `assessment_log` table.
# MAGIC
# MAGIC Known issues:
# MAGIC - Review UI logs feedback on most user actions/clicks.  This unpacker does not merge multiple assessments that orginate from this behavior.
# MAGIC - Every run of this notebook overwrites the existing table.

# COMMAND ----------

from pyspark.sql import DataFrame, SparkSession, window, functions as F, types as T
from typing import Optional, Tuple

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

######################################################################
# Eval dataset schema definitions
######################################################################

RETRIEVAL_OUTPUT_SCHEMA = T.StructType(
    [
        T.StructField("name", T.StringType()),
        T.StructField("chunks", T.ArrayType(CHUNK_SCHEMA)),
    ]
)

GROUND_TRUTH_OUTPUT_SCHEMA = T.StructType(
    [
        T.StructField("content", T.StringType()),
    ]
)

GROUND_TRUTH_SCHEMA = T.StructType(
    [
        T.StructField("text_output", GROUND_TRUTH_OUTPUT_SCHEMA),
        T.StructField("retrieval_output", RETRIEVAL_OUTPUT_SCHEMA),
    ]
)

EVAL_DATASET_INPUT_SCHEMA = T.StructType(
    [
        T.StructField("request", REQUEST_SCHEMA),
        T.StructField("ground_truth", GROUND_TRUTH_SCHEMA),
    ]
)

EVAL_DATASET_OUTPUT_SCHEMA = T.StructType(
    [
        T.StructField("request", REQUEST_SCHEMA),
        T.StructField("trace", TRACE_SCHEMA),
        T.StructField(
            "output",
            T.StructType([T.StructField("choices", CHOICES_SCHEMA, False)]),
            False,
        ),
        T.StructField("ground_truth", GROUND_TRUTH_SCHEMA),
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



# COMMAND ----------

dbutils.widgets.text(
    "inference_table_uc_fqdn",
    label="1. Inference Table UC table",
    defaultValue="catalog.schema.inference_table",
)
inference_table_uc_fqdn = dbutils.widgets.get("inference_table_uc_fqdn")

dbutils.widgets.text(
    "request_log_output_uc_fqdn",
    label="2a. Request Log output UC table",
    defaultValue="catalog.schema.request_log",
)
request_log_output_uc_fqdn = dbutils.widgets.get("request_log_output_uc_fqdn")

dbutils.widgets.text(
    "assessment_log_output_uc_fqdn",
    label="2b. Assessment Log output UC table",
    defaultValue="catalog.schema.assessment_log",
)
assessment_log_output_uc_fqdn = dbutils.widgets.get("assessment_log_output_uc_fqdn")

# COMMAND ----------

# DBTITLE 1,Unpack the payloads

payload_df = spark.table(inference_table_uc_fqdn)
request_logs, assessment_logs = unpack_and_split_payloads(payload_df)

# COMMAND ----------

display(request_logs)

# COMMAND ----------

display(assessment_logs)

# COMMAND ----------

request_logs.write.format("delta").option("mergeSchema", "true").mode(
    "overwrite"
).saveAsTable(request_log_output_uc_fqdn)
assessment_logs.write.format("delta").option("mergeSchema", "true").mode(
    "overwrite"
).saveAsTable(assessment_log_output_uc_fqdn)

# COMMAND ----------

def get_table_url(table_fqdn):
    split = table_fqdn.split(".")
    url = f"https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url


print(f"Wrote `request_log` to: {get_table_url(request_log_output_uc_fqdn)}")
print(f"Wrote `assessment_log` to: {get_table_url(assessment_log_output_uc_fqdn)}")
