# Databricks notebook source
# DBTITLE 1,Databricks RAG Evaluation Installer
# MAGIC %run ./wheel_installer

# COMMAND ----------

# Get trace as JSON

# COMMAND ----------

# DBTITLE 1,Get Trace from Chain
from databricks.rag.scoring import predictions as rag
from databricks import rag_eval

def _convert_trace_buffer_to_trace_object(trace_buffer):
    """Convert the trace buffer from the callback handler to the Trace object."""
    if len(trace_buffer) == 0:
        start_timestamp = None
        end_timestamp = None
    elif len(trace_buffer) == 1:
        start_timestamp = trace_buffer[0].start_timestamp
        end_timestamp = trace_buffer[0].end_timestamp
    else:
        start_timestamp = trace_buffer[0].start_timestamp
        end_timestamp = trace_buffer[-1].end_timestamp
    return rag.Trace(
        steps=trace_buffer,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

def experimental_get_json_trace(langchain_model, model_input):
  handler = rag.DatabricksChainCallback()
  langchain_wrapped_model = rag._LangChainModelWrapper(langchain_model)
  # noinspection PyTypeChecker
  model_output: dict = langchain_wrapped_model._predict_with_callbacks(
      model_input, callback_handlers=[handler], convert_chat_responses=True
  )
  trace_buffer = handler.trace_buffer
  json_trace = _convert_trace_buffer_to_trace_object(trace_buffer).to_json_obj()

  return json_trace

# COMMAND ----------

# DBTITLE 1,Log YAML config to MLflow run
# Log YAML parameters to the MLflow run

# COMMAND ----------

# DBTITLE 1,Config Params MLflow Logger
from databricks.rag import RagConfig
import mlflow

def log_to_mlflow_run(self, run_id):
    config = self._read_config()
    with mlflow.start_run(run_id=run_id):
      mlflow.log_params(config)
      # for key, value in config.items():
      #   print(f'{key}: {value}')
      #   mlflow.log_param(key, value)

RagConfig.experimental_log_to_mlflow_run = log_to_mlflow_run

# COMMAND ----------

# DBTITLE 1,Evaluation Metrics Tracker
# Add eval metrics to Mlflow

# COMMAND ----------

import json
import yaml
import mlflow

client = mlflow.tracking.MlflowClient()

METRIC_NAMES = [
    "response_metrics.token_count",
    "response_metrics.exact_match",
    "response_metrics.llm_judged_harmful",
    "response_metrics.llm_judged_faithful_to_context",
    "response_metrics.llm_judged_relevant_to_question_and_context",
    "response_metrics.llm_judged_relevant_to_question",
    "response_metrics.llm_judged_answer_good",
    "retrieval_metrics.ground_truth_precision_at_1",
    "retrieval_metrics.ground_truth_recall_at_1",
    "retrieval_metrics.ground_truth_ndcg_at_1",
    "retrieval_metrics.ground_truth_precision_at_3",
    "retrieval_metrics.ground_truth_recall_at_3",
    "retrieval_metrics.ground_truth_ndcg_at_3",
    "retrieval_metrics.ground_truth_precision_at_5",
    "retrieval_metrics.ground_truth_recall_at_5",
    "retrieval_metrics.ground_truth_ndcg_at_5",
    "retrieval_metrics.ground_truth_precision_at_10",
    "retrieval_metrics.ground_truth_recall_at_10",
    "retrieval_metrics.ground_truth_ndcg_at_10",
    "retrieval_metrics.llm_judged_precision_at_1",
    "retrieval_metrics.llm_judged_precision_at_3",
    "retrieval_metrics.llm_judged_precision_at_5",
    "retrieval_metrics.llm_judged_precision_at_10",
]
NAME_MAP = {
    "response_metrics.token_count": "response_metrics/token_count",
    "response_metrics.exact_match": "response_metrics/exact_match",
    "response_metrics.llm_judged_harmful": "response_metrics/llm_judged/harmful",
    "response_metrics.llm_judged_faithful_to_context": "response_metrics/llm_judged/faithful_to_context",
    "response_metrics.llm_judged_relevant_to_question_and_context": "response_metrics/llm_judged/relevant_to_question_and_context",
    "response_metrics.llm_judged_relevant_to_question": "response_metrics/llm_judged/relevant_to_question",
    "response_metrics.llm_judged_answer_good": "response_metrics/llm_judged/answer_correct",
    "retrieval_metrics.ground_truth_precision_at_1": "retrieval_metrics/ground_truth/k_1/precision",
    "retrieval_metrics.ground_truth_recall_at_1": "retrieval_metrics/ground_truth/k_1/recall",
    "retrieval_metrics.ground_truth_ndcg_at_1": "retrieval_metrics/ground_truth/k_1/ndcg",
    "retrieval_metrics.ground_truth_precision_at_3": "retrieval_metrics/ground_truth/k_3/precision",
    "retrieval_metrics.ground_truth_recall_at_3": "retrieval_metrics/ground_truth/k_3/recall",
    "retrieval_metrics.ground_truth_ndcg_at_3": "retrieval_metrics/ground_truth/k_3/ndcg",
    "retrieval_metrics.ground_truth_precision_at_5": "retrieval_metrics/ground_truth/k_5/precision",
    "retrieval_metrics.ground_truth_recall_at_5": "retrieval_metrics/ground_truth/k_5/recall",
    "retrieval_metrics.ground_truth_ndcg_at_5": "retrieval_metrics/ground_truth/k_5/ndcg",
    "retrieval_metrics.ground_truth_precision_at_10": "retrieval_metrics/ground_truth/k_10/precision",
    "retrieval_metrics.ground_truth_recall_at_10": "retrieval_metrics/ground_truth/k_10/recall",
    "retrieval_metrics.ground_truth_ndcg_at_10": "retrieval_metrics/ground_truth/k_10/ndcg",
    "retrieval_metrics.llm_judged_precision_at_1": "retrieval_metrics/llm_judged/k_1/precision",
    "retrieval_metrics.llm_judged_precision_at_3": "retrieval_metrics/llm_judged/k_1/precision",
    "retrieval_metrics.llm_judged_precision_at_5": "retrieval_metrics/llm_judged/k_1/precision",
    "retrieval_metrics.llm_judged_precision_at_10": "retrieval_metrics/llm_judged/k_1/precision",
}
METRIC_SQL_COMPUTATION_TEMPLATE = (
    "avg(coalesce(cast({metric_type}.{metric_name} as float), 0)) as `{metric_new_name}`"
)
METRIC_SELECT_TEMPLATE = """SELECT
  mlflow_run_url,
  {metric_sql_statement}
FROM
  {table_name}
where
  mlflow_run_url = '{mlflow_run_url}'
group by
  1"""


def experimental_add_metrics_to_run(eval_results, run_id=None):
    if run_id is None:
        run_id = eval_results.mlflow_run_id
    metric_sqls = [
        METRIC_SQL_COMPUTATION_TEMPLATE.format(
            metric_type=metric.split(".")[0], metric_name=metric.split(".")[1], metric_new_name=NAME_MAP[metric]
        )
        for metric in METRIC_NAMES
    ]
    metric_sql_statement = ", ".join(metric_sqls)
    sql_to_run = METRIC_SELECT_TEMPLATE.format(
        metric_sql_statement=metric_sql_statement,
        table_name=eval_results.eval_metrics_table_name,
        mlflow_run_url=eval_results.mlflow_run_url,
    )
    # print(sql_to_run)
    df = spark.sql(sql_to_run)

    agg_metrics = json.loads(df.toJSON().collect()[0])

    for metric_name in agg_metrics.keys():
        if metric_name != "mlflow_run_url":
            # print( + str())
            client.log_metric(
                run_id, metric_name, agg_metrics[metric_name], synchronous=True
            )


EVAL_DATA_SQL_TEMPLATE = """select
  a.request_id,
  a.app_version,
  a.request,
  a.expected_response,
  a.response,
  a.expected_retrieval_context,
  a.retrieval_context,
  b.response_assessment.ratings.harmful as judge_harmful,
  b.response_assessment.ratings.faithful_to_context as judge_faithful_to_context,
  b.response_assessment.ratings.relevant_to_question_and_context as judge_relevant_to_question_and_context,
    b.response_assessment.ratings.relevant_to_question as judge_relevant_to_question,
  b.response_assessment.ratings.answer_good as judge_answer_good,
  b.retrieval_assessment.positional_ratings as judge_context_relevant_to_question
FROM
  {eval_metrics_table_name} a
  left join {assessments_table_name} b on a.request_id = b.request_id
  and a.app_version = b.app_version
where
  a.mlflow_run_url = '{mlflow_run_url}'"""


def experimental_add_eval_outputs_to_run(eval_results, run_id=None):
    if run_id is None:
        run_id = eval_results.mlflow_run_id
    sql = EVAL_DATA_SQL_TEMPLATE.format(
        eval_metrics_table_name=eval_results.eval_metrics_table_name,
        assessments_table_name=eval_results.assessments_table_name,
        mlflow_run_url=eval_results.mlflow_run_url,
    )
    df = spark.sql(sql)
    json_data_table = df.toJSON().collect()
    json_data_table = [json.loads(row) for row in json_data_table]

    # print(json_data_table)

    dict_for_mlflow_logging = {
        key: [str(d.get(key, "null")) for d in json_data_table]
        for key in json_data_table[0]
    }

    client.log_table(
        run_id,
        data=dict_for_mlflow_logging,
        artifact_file="databricks_eval_results.json",
    )


_RAG_EVAL_PREFIX = "rag_eval"


def experimental_add_eval_config_tags_to_run(eval_results, config=None, run_id=None):
    if run_id is None:
        run_id = eval_results.mlflow_run_id
    if config is not None:
      try:
          config_json = yaml.safe_load(config)
      except Exception as e:
          print("Error parsing eval config, not adding this tag to the run")
          config_json = "!Error!"
    else:
      config_json = "Unknown configuration"
    tags = {
        f"{_RAG_EVAL_PREFIX}.dashboard_url": eval_results.dashboard_url,
        f"{_RAG_EVAL_PREFIX}.eval_metrics_table": eval_results.eval_metrics_table_name,
        f"{_RAG_EVAL_PREFIX}.assessments_table": eval_results.assessments_table_name,
        f"{_RAG_EVAL_PREFIX}.eval_set_table": eval_results.eval_metrics_table_name,
        f"{_RAG_EVAL_PREFIX}.config": config_json,
    }

    for key, value in tags.items():
        client.set_tag(run_id, key, value)
        # print(key)
        # print(value)
