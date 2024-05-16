### Tutorial 4: Running evaluation on an existing RAG chain / app built outside of RAG Studio

This tutorial walks you through using Evaluation Suite to evaluate the quality of a RAG chain built outside of RAG Studio or already deployed outside of Databricks.
Open the [notebook](4_evaluation_of_existing_chains.py) to get started.

Evaluation Harness supports 2 approaches to provide your chain’s outputs in order to generate quality/cost/latency Metrics.   This tutorial shows you approach #2.

| Approach | Description | When to use |
| --- | --- | --- |
| 1 | **Eval Harness runs a chain on your behalf.** Pass a reference to the chain itself so Evaluation Harness can generate the outputs on your behalf | - Your chain is logged using MLflow w/ MLflow Tracing enabled <br> - Your chain is available as a Python function in your local notebook |
| 2 | **Run chain yourself, pass outputs to Eval Harness.** Run the chain being evaluated yourself, capturing the chain’s outputs and passing the outputs as a Pandas DataFrame | - Your chain is developed outside of Databricks <br> - You want to evaluate outputs from a chain already running in production <br> - You are testing different evaluation / LLM Judge configurations and your chain doesn’t produce deterministic outputs (e.g., LLM has a high temperature) |

#### Step 1: Gather your chain's outputs (and optionally ground truth) to evaluate
First, you need to ETL your chain's logs into the Evaluation Harness input schema. `evaluate(...)` takes a parameter `data` that is a Pandas DataFrame containing your chain's outputs and, optionally, your Evaluation Set w/ ground truth.  Let's look at a few examples.

*- For full details on the schema, view the `Evaluation Harness Input Schema` section of the RAG Studio documentation.*<br/>
*- For full details on the metrics available, view the `LLM Judges & Metrics` section of the RAG Studio documentation.*

These Dictionary-based examples are provided just to show the schema. You do NOT have to start from a Dictionary - you can use any existing Pandas or Spark DataFrame with this schema.

Below, we walk you through the 5 levels of data that you may have available.  As you increase your data level, Evaluation Suite can offer you additional functionality
|                                             | Level 1 | Level 2 | Level 3 | Level 4 |
|-----------------------------------------------------------------------|---------|---------|---------|---------|
| **Required data**                                     |        |        |        |        |
| Chain logs: `request`, `response`                                     | ✔       | ✔       | ✔       | ✔       |
| Chain logs: `retrieved_context`                                       | X       | ✔       | ✔       | ✔       |
| Evaluation set: `expected_response`                                   | X       | X       | ✔       | ✔       |
| Evaluation set: `expected_retrieved_context`                          | X       | X       | X       | ✔       |
| **Supported metrics**                                     |        |        |        |        |
| `response/llm_judged/relevance_to_query_rating/average`               | ✔       | ✔       | ✔       | ✔       |
| `response/llm_judged/harmfulness_rating/average`                      | ✔       | ✔       | ✔       | ✔       |
| `chain/request_token_count`                                           | ✔       | ✔       | ✔       | ✔       |
| `chain/response_token_count`                                          | ✔       | ✔       | ✔       | ✔       |
| Customer-defined LLM judges                                           | ✔       | ✔       | ✔       | ✔       |
| `retrieval/llm_judged/chunk_relevance_precision/average`              | X       | ✔       | ✔       | ✔       |
| `response/llm_judged/groundedness_rating/average`                     | X       | ✔       | ✔       | ✔       |
| `response/llm_judged/correctness_rating/average`                      | X       | X       | ✔       | ✔       |
| `retrieval/ground_truth/document_recall/average`                      | X       | X       | X       | ✔       |
| `retrieval/ground_truth/document_precision/average`                   | X       | X       | X       | ✔       |


##### Level 1: Chain's request & response 
```
level_1_data = [
    {
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "response": "reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.",
    }]
```
##### Level 2: Complete chain logs w/ retrieval info
```
level_2_data = [
    {
        "request_id": "your-request-id", # optional, but useful for tracking
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "response": "reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.",
        "retrieved_context": [
            {
                # In `retrieved_context`, `content` is optional, but DOES deliver additional functionality if provided (the Databricks Context Relevance LLM Judge will run to check the `content`'s relevance to the `request`).
                "content": "reduceByKey reduces the amount of data shuffled by merging values before shuffling.",
                "doc_uri": "doc_uri_2_1",
            },
            {
                "content": "groupByKey may lead to inefficient data shuffling due to sending all values across the network.",
                "doc_uri": "doc_uri_6_extra",
            },
        ],
    }]
```

##### Level 3: Complete chain logs w / labeled ground truth answer
```
level_3_data  = [
    {
        "request_id": "your-request-id", 
        "request": "What is the difference between reduceByKey and groupByKey in Spark?",
        "expected_response": "There's no significant difference.",
        "response": "reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.",
        "retrieved_context": [
            {
                # In `retrieved_context`, `content` is optional, but DOES deliver additional functionality if provided (the Databricks Context Relevance LLM Judge will run to check the `content`'s relevance to the `request`).
                "content": "reduceByKey reduces the amount of data shuffled by merging values before shuffling.",
                "doc_uri": "doc_uri_2_1",
            },
            {
                "content": "groupByKey may lead to inefficient data shuffling due to sending all values across the network.",
                "doc_uri": "doc_uri_6_extra",
            },
        ],
    }]
```

##### Level 4: Complete chain logs w/ labeled ground truth answer & retrieval context

```
level_4_data  = [
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
        "response": "reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.",
        "retrieved_context": [
            {
                # In `retrieved_context`, `content` is optional, but DOES deliver additional functionality if provided (the Databricks Context Relevance LLM Judge will run to check the `content`'s relevance to the `request`).
                "content": "reduceByKey reduces the amount of data shuffled by merging values before shuffling.",
                "doc_uri": "doc_uri_2_1",
            },
            {
                "content": "groupByKey may lead to inefficient data shuffling due to sending all values across the network.",
                "doc_uri": "doc_uri_6_extra",
            },
        ],
    }]
```

#### Step 2: Run evaluation

Once you have your data available, running evaluation is simple single-line command. 

Here we use the `level_4_data_df` but you can replace this with any of the other example DataFrames above. 

```
import pandas as pd
import mlflow
# If you do not start a MLflow run, `evaluate(...) will start a Run on your behalf.
level_4_data_df = pd.DataFrame(level_4_data)
with mlflow.start_run(run_name="level_4_data"):
  evaluation_results = mlflow.evaluate(data=level_4_data_df, model_type="databricks-rag")
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