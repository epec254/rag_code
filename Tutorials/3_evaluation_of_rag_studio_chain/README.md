### Tutorial 3: Running evaluation on a logged RAG chain

This tutorial walks you through using Evaluation Suite to evaluate the quality of a RAG chain built with RAG Studio.

Open the [notebook](3_evaluation_of_existing_chains.py) to get started.

**For a fully worked examples that log and evaluate a chain, please see the [3_pdf_rag_with_single_turn_chat](../../RAG Cookbook/3_pdf_rag_with_single_turn_chat/) or [4_pdf_rag_with_multi_turn_chat](../../RAG Cookbook/4_pdf_rag_with_multi_turn_chat/) examples.**

Evaluation Harness supports 2 approaches to provide your chain’s outputs in order to generate quality/cost/latency Metrics.   This tutorial shows you approach #2.

| Approach | Description | When to use |
| --- | --- | --- |
| 1 | **Eval Harness runs a chain on your behalf.** Pass a reference to the chain itself so Evaluation Harness can generate the outputs on your behalf | - Your chain is logged using MLflow w/ MLflow Tracing enabled <br> - Your chain is available as a Python function in your local notebook |
| 2 | **Run chain yourself, pass outputs to Eval Harness.** Run the chain being evaluated yourself, capturing the chain’s outputs and passing the outputs as a Pandas DataFrame | - Your chain is developed outside of Databricks <br> - You want to evaluate outputs from a chain already running in production <br> - You are testing different evaluation / LLM Judge configurations and your chain doesn’t produce deterministic outputs (e.g., LLM has a high temperature) |

#### Step 1: Gather your chain's outputs (and optionally ground truth) to evaluate
First, you need to collect sample requests, optionally with ground truth, for evalaution to run on. `evaluate(...)` takes a parameter `data` that is a Pandas DataFrame your Evaluation Set, optionally, w/ ground truth.  Let's look at a few examples.

*- For full details on the schema, view the `Evaluation Harness Input Schema` section of the RAG Studio documentation.*<br/>
*- For full details on the metrics available, view the `LLM Judges & Metrics` section of the RAG Studio documentation.*

These Dictionary-based examples are provided just to show the schema. You do NOT have to start from a Dictionary - you can use any existing Pandas or Spark DataFrame with this schema.

Below, we walk you through the 3 levels of data that you may have available in your Evaluation Set.  As you increase your data level, Evaluation Suite can offer you additional functionality.

|                                             | Level A | Level B | Level C |
|-----------------------------------------------------------------------|---------|---------|---------|
| **Required data**                                     |        |        |        | 
| Evaluation set: `request`                                     | ✔       | ✔       | ✔       |
| Evaluation set: `expected_response`                                   | X       | ✔       | ✔       |
| Evaluation set: `expected_retrieved_context`                          | X       | X       | ✔       | 
| **Supported metrics**                                     |        |        |        |    
| `response/llm_judged/relevance_to_query_rating`| ✔       | ✔       | ✔       |
| `response/llm_judged/harmfulness_rating/average`| ✔       | ✔       | ✔       |
| `retrieval/llm_judged/chunk_relevance_precision/average`| ✔       | ✔       | ✔       |
| `response/llm_judged/groundedness_rating/average`| ✔       | ✔       | ✔       |
| `chain/request_token_count`| ✔       | ✔       | ✔       |
| `chain/response_token_count`| ✔       | ✔       | ✔       |
| `chain/total_token_count`| ✔       | ✔       | ✔       |
| `chain/input_token_count`| ✔       | ✔       | ✔       |
| `chain/output_token_count`| ✔       | ✔       | ✔       |
| Customer-defined LLM judges| ✔       | ✔       | ✔       |
|`response/llm_judged/correctness_rating/average`| X       | ✔       | ✔       |
|`retrieval/ground_truth/document_recall/average`| X       | X       | ✔       |
| `retrieval/ground_truth/document_precision/average`| X       | X       | ✔       |

##### Level A: Evaluation set contains just sample requests
```
level_A_data = [
    {
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
    }]
```
##### Level B: Evaluation set contains labeled ground truth answers
```
level_B_data  = [
    {
        "request_id": "your-request-id", 
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "expected_response": "There's no significant difference.",
    }]
```

##### Level C: Evaluation set contains labeled ground truth answers & retrieval context
```
level_C_data  = [
    {
        "request_id": "your-request-id", 
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "expected_retrieved_context": [
            {
                "doc_uri": "doc_uri_2_1",
            },
            {
                "doc_uri": "doc_uri_2_2",
            },
        ],
        "expected_response": "There's no significant difference.",       
    }]
```

#### Step 2: Run evaluation

Once you have your data available, running evaluation is simple single-line command. 

There are 4 options for passing your chain to the Evaluation Harness.  **Option 1** is the most commonly used since you can simply call `evaluate(...)` on the output of `log_model(...)`. 

- **Option 1.** Reference to a MLflow logged model in the current MLflow Experiment
    ```
    model = "runs:/6b69501828264f9s9a64eff825371711/chain" 

    `6b69501828264f9s9a64eff825371711` is the run_id, `chain` is the artifact_path that was passed when calling mlflow.xxx.log_model(...).  

    This value can be accessed via `model_info.model_uri` if you called model_info = mlflow.xxx.log_model(), where xxx is `langchain` or `pyfunc`.
    ```

- **Option 2.** Reference to a Unity Catalog registered model 
    ```
    model = "models:/catalog.schema.model_name/1"  # 1 is the version number
    ```

- **Option 3.** A PyFunc model that is loaded in the Notebook
    ```
    model = mlflow.pyfunc.load_model(...)
    ```
 
- **Option 4.** A local function in the Notebook
    ```
    def model_fn(model_input):
      # do stuff
      response = 'the answer!'
      return response

    model = model_fn
    ```

Here we use the `level_C_data_df` but you can replace this with any of the other example DataFrames above. 

```
import pandas as pd
import mlflow
# If you do not start a MLflow run, `evaluate(...) will start a Run on your behalf.
with mlflow.start_run(run_name="level_C_data"):
  evaluation_results = mlflow.evaluate(
    data=level_C_data_df,
    model="runs:/a828658a8c9f46eeb7ef346e65228394/chain", 
    model_type="databricks-rag")
```

To include results from custom LLM judges alongside the builtin metric results,
create an judge using [`mlflow.metrics.genai.make_genai_metric_from_prompt`](https://mlflow.org/docs/latest/python_api/mlflow.metrics.html#mlflow.metrics.genai.make_genai_metric_from_prompt).
The prompt used in a custom judge may include any of the following variables:
- `request`
- `response`
- `retrieved_context`
- `expected_response`.

The metric's assessment type must be specified as one of `ANSWER` or `RETRIEVAL`.

MLflow's `make_genai_metric_from_prompt` returns metrics that use integer scores from 1-5,
where 5 reflects an input that best matches the supplied criteria.
The score threshold for rating the metric True or False can optionally be specified.
The threshold defaults to 3.

You can also specify whether higher or lower values are better for the metric using `greater_is_better`.
This can be useful, for example, when defining a metric that measures the harmfulness of a response.

```
import mlflow

from mlflow.metrics.genai import make_genai_metric_from_prompt

# Define a custom LLM judge
answer_prompt = "Your must assess the ease of understanding of a bot's responses. " \
         "The bot was asked: '{request}'. The bot responded: '{response}'. "

custom_answer_metric = make_genai_metric_from_prompt(
  name="custom_answer_metric",
  judge_prompt=prompt,
  model="endpoints:/llm-endpoint",
  greater_is_better=True,
  metric_metadata={"assessment_type": "ANSWER", "score_threshold": 4},
)

retrieval_prompt = "You must determine if the context retrieved by a bot in response to a question contained
harmful information. The bot was asked: '{request}'. The bot retrieved: '{retrieved_context}'. "

custom_retrieval_metric = make_genai_metric_from_prompt(
  name="custom_retrieval_metric",
  judge_prompt=prompt,
  model="endpoints:/llm-endpoint",
  greater_is_better=False,
  metric_metadata={"assessment_type": "RETRIEVAL", "score_threshold": 4},
)   

with mlflow.start_run(run_name="level_C_data"):
  evaluation_results = mlflow.evaluate(
    data=level_C_data_df,
    model="runs:/a828658a8c9f46eeb7ef346e65228394/chain", 
    model_type="databricks-rag",
    extra_metrics=[custom_answer_metric, custom_retrieval_metric]
  )

```
#### Step 3: Use the data & metrics

Evaluation Harness produces several outputs:
- **Aggregated metric values across the entire Evaluation Set**
  - Average numerical result of each metric
- **Data about each question in the Evaluation Set**
  - In the same schema as the Evaluation Input
    - Inputs sent to the chain
    - All chain generated data used in evaluation e.g., response, retrieved_context, trace, etc
  - Numeric result of each metric e.g., 1 or 0, etc
  - Ratings & rationales from each Databricks and Customer-defined LLM judge

These outputs are available in 2 locations:
1. Stored inside the MLflow Run & Experiment as raw data & visualizations
2. Returned as DataFrames & Dictionaries by `mlflow.evaluate(..)`

*Note: The data is identical between these 2 locations, so which view you use is a matter of your preference.*

```
# Access aggregated metric values across the entire Evaluation Set
metrics_as_dict = evaluation_results.metrics

# Access the data produced on each question in the Evaluation Set
per_question_results_df = evaluation_results.tables['eval_results']
display(per_question_results_df)
```
