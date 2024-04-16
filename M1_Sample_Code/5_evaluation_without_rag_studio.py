# Databricks notebook source
# MAGIC %md
# MAGIC **This notebook shows you how to use Databricks' Evaluation Suite *without* RAG studio e.g., for a chain that you have deployed already.**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Metrics & LLM Judges overview
# MAGIC Databricks provides a set of metrics that enable you to measure the quality, cost and latency of your RAG app. These metrics are curated by Databricks' Research team as the most relevant (no pun intended) metrics for evaluating RAG applications.
# MAGIC
# MAGIC RAG metrics can be computed using either:
# MAGIC 1. Human-labeled ground truth assessments
# MAGIC 2. LLM judge-labeled assessments 
# MAGIC
# MAGIC A subset of the metrics work only with *either* LLM judge-labeled OR human-labeled ground truth asessments.
# MAGIC
# MAGIC ## Unstructured docs retrieval & generation metrics
# MAGIC
# MAGIC ### Retriever
# MAGIC
# MAGIC RAG Studio supports the following metrics for evaluating the retriever.
# MAGIC
# MAGIC | Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments | 
# MAGIC |-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
# MAGIC | Are the retrieved chunks relevant to the user’s query?                            | Precision of "relevant chunk" @ K | 0 to 100% | 0 to 100% | ✔️ | ✔️ |
# MAGIC | Are **ALL** chunks that are relevant to the user’s query retrieved?               | Recall of "relevant chunk" @ K | 0 to 100% |0 to 100% | ✔️ |✖️ |
# MAGIC | Are the retrieved chunks returned in the correct order of most to least relevant? | nDCG of "relevant chunk" @ K | 0 to 1 | 0 to 1 |✔️ | ✖️ |
# MAGIC
# MAGIC ### Generation model
# MAGIC
# MAGIC These metrics measure the generation model's performance when the prompt is augemented with unstructured docs from a retrieval step.
# MAGIC
# MAGIC | Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments | 
# MAGIC |-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
# MAGIC | Is the LLM not hallucinating & responding based ONLY on the context provided? | Faithfulness (to context) | true/false | 0 to 100% | ✖️ | ✔️ |
# MAGIC | Is the response on-topic given the query AND retrieved contexts? | Answer relevance (to query given the context) | true/false | 0 to 100% | ✖️ | ✔️ | 
# MAGIC | Is the response on-topic given the query? | Answer relevance (to query) | true/false | 0 to 100% | ✖️ | ✔️ | 
# MAGIC | What is the cost of the generation? | Token Count | sum(tokens) | sum(tokens) | n/a |n/a |
# MAGIC | What is the latency of generation? | Latency | milliseconds | average(milliseconds) | n/a | n/a |
# MAGIC
# MAGIC ### RAG chain metrics
# MAGIC
# MAGIC These metrics measure the chain's final response back to the user.  
# MAGIC
# MAGIC | Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments | 
# MAGIC |-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
# MAGIC | Is the response accurate (correct)? | Answer correctness (vs. ground truth) | true/false | 0 to 100% |✔️| ✖️ |
# MAGIC | Does the response violate any of my company policies (racism, toxicity, etc)? | Toxicity | true/false | 0 to 100% | ✖️ | ✔️|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup and sample data

# COMMAND ----------

# DBTITLE 1,Install Evaluation Suite
# MAGIC %run ./wheel_installer

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./RAG_Experimental_Code

# COMMAND ----------

# DBTITLE 1,Import Evaluation Suite
from databricks import rag_eval
import yaml

# COMMAND ----------

# DBTITLE 1,Config
dbutils.widgets.text("catalog", defaultValue="catalog")
dbutils.widgets.text("schema", defaultValue="schema")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")


# COMMAND ----------

# MAGIC %md
# MAGIC This cell is very long since it contains the sample data.  You do not need to read the contents of this cell.

# COMMAND ----------

# DBTITLE 1,Sample Data
SAMPLE_DATASETS = [
    "spark_bot_eval_dataset",  # eval dataset with only questions and no ground truth
    "spark_bot_eval_dataset_with_ground_truth",  # eval dataset that includes the ground truth for all questions and all fields filled out
    "spark_bot_v1_answer_sheet",  # v1 answer sheet with all fields filled out
    "spark_bot_v1_answer_sheet_minimal",  # v1 answer sheet without retrieval_context
    "spark_bot_v2_answer_sheet",  # v2 answer sheet with all fields filled out
    "spark_bot_v2_answer_sheet_minimal",  # v2 answer sheet without retrieval_context
    "spark_bot_v3_answer_sheet",  # v3 answer sheet that contains all wrong answers
    "spark_bot_v3_answer_sheet_minimal",  # v3 minimal answer sheet that contains all wrong answers
]

RAW_DATA = {'spark_bot_eval_dataset': [{'request_id': 'd2f19cb5-8d15-4e5a-a131-800b49d7c28f', 'request': 'What is the difference between reduceByKey and groupByKey in Spark?'}, {'request_id': '60ed4657-5acc-4960-a25b-c91969b28c2c', 'request': 'How can you minimize data shuffling in Spark?'}, {'request_id': '2e8d7aba-a1ab-4442-a404-cc334e57fee1', 'request': 'Explain broadcast variables in Spark. How do they enhance performance?'}, {'request_id': '0ce5ddf0-23aa-4295-b950-67cbf0490574', 'request': 'What are the main components of Sparks execution model?'}, {'request_id': 'f6ed004d-5df1-4eb8-b0dc-13c37180543a', 'request': 'How does Spark achieve fault tolerance?'}, {'request_id': 'f7be7cc6-264e-4cc8-83ef-b61903eb9b8c', 'request': 'Describe the role of the Spark Driver.'}, {'request_id': 'bf4e11df-3085-4863-97d5-d9577fe0b387', 'request': 'What are stages in Spark execution?'}, {'request_id': 'e3bdaab0-7996-485d-8ed8-e976439ca4b2', 'request': 'How can you manage data skew in Spark?'}, {'request_id': 'a8f733a1-205c-4e5d-8a16-23e81ddc2505', 'request': 'What is the benefit of using DataFrames over RDDs?'}, {'request_id': 'e08d1d26-ff11-406f-a6be-70712e1b7347', 'request': 'How do you increase the number of executors in Apache Spark?'}], 'spark_bot_eval_dataset_with_ground_truth': [{'request_id': 'd2f19cb5-8d15-4e5a-a131-800b49d7c28f', 'request': 'What is the difference between reduceByKey and groupByKey in Spark?', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_2_1', 'content': 'Answer segment 1 related to What is the difference between reduceByKey and groupByKey in Spark?'}, {'doc_uri': 'doc_uri_2_2', 'content': 'Answer segment 2 related to What is the difference between reduceByKey and groupByKey in Spark?'}], 'expected_response': 'Theres no significant difference.'}, {'request_id': '60ed4657-5acc-4960-a25b-c91969b28c2c', 'request': 'How can you minimize data shuffling in Spark?', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_3_1', 'content': 'Answer segment 1 related to How can you minimize data shuffling in Spark?'}, {'doc_uri': 'doc_uri_3_2', 'content': 'Answer segment 2 related to How can you minimize data shuffling in Spark?'}], 'expected_response': 'Minimize by using narrow transformations.'}, {'request_id': '2e8d7aba-a1ab-4442-a404-cc334e57fee1', 'request': 'Explain broadcast variables in Spark. How do they enhance performance?', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_4_1', 'content': 'Answer segment 1 related to Explain broadcast variables in Spark. How do they enhance performance?'}, {'doc_uri': 'doc_uri_4_2', 'content': 'Answer segment 2 related to Explain broadcast variables in Spark. How do they enhance performance?'}], 'expected_response': 'They dont enhance performance.'}, {'request_id': '0ce5ddf0-23aa-4295-b950-67cbf0490574', 'request': 'What are the main components of Sparks execution model?', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_5_1', 'content': 'Answer segment 1 related to What are the main components of Sparks execution model?'}, {'doc_uri': 'doc_uri_5_2', 'content': 'Answer segment 2 related to What are the main components of Sparks execution model?'}], 'expected_response': 'Driver, executors, and a cluster manager.'}, {'request_id': 'f6ed004d-5df1-4eb8-b0dc-13c37180543a', 'request': 'How does Spark achieve fault tolerance?', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_1_1', 'content': 'Spark achieves fault tolerance through its RDD abstraction, allowing for recomputation of lost data.'}, {'doc_uri': 'doc_uri_1_2', 'content': 'Fault tolerance in Spark is managed by tracking the lineage of each RDD so that it can be rebuilt in case of data loss.'}], 'expected_response': 'Spark achieves fault tolerance through its RDD abstraction, allowing for recomputation of lost data. Fault tolerance in Spark is managed by tracking the lineage of each RDD so that it can be rebuilt in case of data loss.'}, {'request_id': 'f7be7cc6-264e-4cc8-83ef-b61903eb9b8c', 'request': 'Describe the role of the Spark Driver.', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_2_1', 'content': 'The Spark Driver program runs the main() method of the application and is the point of control of the Spark job.'}, {'doc_uri': 'doc_uri_2_2', 'content': 'The Driver is responsible for converting the user program into tasks and scheduling them to run on executors.'}], 'expected_response': 'The Spark Driver program runs the main() method of the application and is the point of control of the Spark job. The Driver is responsible for converting the user program into tasks and scheduling them to run on executors.'}, {'request_id': 'bf4e11df-3085-4863-97d5-d9577fe0b387', 'request': 'What are stages in Spark execution?', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_3_1', 'content': 'Stages in Spark are a set of tasks that share the same computation, but each operates on a different subset of data.'}, {'doc_uri': 'doc_uri_3_2', 'content': 'A stage is completed when all tasks within it have completed. Stages are divided by actions or shuffles.'}], 'expected_response': 'Stages in Spark are a set of tasks that share the same computation, but each operates on a different subset of data. A stage is completed when all tasks within it have completed. Stages are divided by actions or shuffles.'}, {'request_id': 'e3bdaab0-7996-485d-8ed8-e976439ca4b2', 'request': 'How can you manage data skew in Spark?', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_4_1', 'content': 'Data skew can be managed by repartitioning the data more evenly across the cluster.'}, {'doc_uri': 'doc_uri_4_2', 'content': 'Salting techniques can also be used to distribute skewed data more evenly across partitions.'}], 'expected_response': 'Data skew can be managed by repartitioning the data more evenly across the cluster. Salting techniques can also be used to distribute skewed data more evenly across partitions.'}, {'request_id': 'a8f733a1-205c-4e5d-8a16-23e81ddc2505', 'request': 'What is the benefit of using DataFrames over RDDs?', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_5_1', 'content': 'DataFrames allow for optimizations through the Catalyst optimizer, which can optimize query execution plans.'}, {'doc_uri': 'doc_uri_5_2', 'content': 'Using DataFrames over RDDs provides a higher level of abstraction, making it easier to work with structured data.'}], 'expected_response': 'DataFrames allow for optimizations through the Catalyst optimizer, which can optimize query execution plans. Using DataFrames over RDDs provides a higher level of abstraction, making it easier to work with structured data.'}, {'request_id': 'e08d1d26-ff11-406f-a6be-70712e1b7347', 'request': 'How do you increase the number of executors in Apache Spark', 'expected_retrieval_context': [{'doc_uri': 'doc_uri_1_1', 'content': 'Answer segment 1 related to How do you increase the number of executors in Apache Spark?'}, {'doc_uri': 'doc_uri_1_2', 'content': 'Answer segment 2 related to How do you increase the number of executors in Apache Spark?'}], 'expected_response': 'Increase the spark.executor.instances property value.'}], 'spark_bot_v1_answer_sheet': [{'request_id': 'd2f19cb5-8d15-4e5a-a131-800b49d7c28f', 'app_version': 'version_name_1', 'response': 'reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.', 'retrieval_context': [{'doc_uri': 'doc_uri_6_1', 'content': 'reduceByKey reduces the amount of data shuffled by merging values before shuffling.'}, {'doc_uri': 'doc_uri_6_extra', 'content': 'groupByKey may lead to inefficient data shuffling due to sending all values across the network.'}]}, {'request_id': '60ed4657-5acc-4960-a25b-c91969b28c2c', 'app_version': 'version_name_1', 'response': 'Minimizing data shuffling in Spark involves using narrow transformations and avoiding operations like groupByKey when possible.', 'retrieval_context': [{'doc_uri': 'doc_uri_7_1', 'content': 'Narrow transformations, like map and filter, minimize shuffling by processing data locally.'}, {'doc_uri': 'doc_uri_7_extra', 'content': 'Choosing operations wisely, such as reduceByKey over groupByKey, can also reduce shuffle.'}]}, {'request_id': '2e8d7aba-a1ab-4442-a404-cc334e57fee1', 'app_version': 'version_name_1', 'response': 'Broadcast variables enhance performance by keeping a read-only variable cached on each machine, rather than shipping a copy of it with tasks.', 'retrieval_context': [{'doc_uri': 'doc_uri_8_1', 'content': 'Broadcast variables reduce data transfer by caching data on executors.'}, {'doc_uri': 'doc_uri_8_extra', 'content': 'They are useful for large datasets that are needed by all nodes for tasks or operations.'}]}, {'request_id': '0ce5ddf0-23aa-4295-b950-67cbf0490574', 'app_version': 'version_name_1', 'response': 'The main components of Spark’s execution model include the Driver, Executors, and the Cluster Manager.', 'retrieval_context': [{'doc_uri': 'doc_uri_9_1', 'content': 'The Driver schedules tasks and communicates with the Cluster Manager.'}, {'doc_uri': 'doc_uri_9_extra', 'content': 'Executors run tasks assigned by the Driver and return results back to it.'}]}, {'request_id': 'e08d1d26-ff11-406f-a6be-70712e1b7347', 'app_version': 'version_name_1', 'response': 'Increasing the number of executors in Spark can be achieved by configuring spark.executor.instances in Spark’s configuration.', 'retrieval_context': [{'doc_uri': 'doc_uri_10_1', 'content': 'Adjusting spark.executor.instances allows for scaling the number of executors.'}, {'doc_uri': 'doc_uri_10_extra', 'content': 'More executors can improve parallelism and performance for distributed tasks.'}]}, {'request_id': 'f6ed004d-5df1-4eb8-b0dc-13c37180543a', 'app_version': 'version_name_1', 'response': 'Spark ensures fault tolerance through its RDDs, which can rebuild data using lineage.', 'retrieval_context': [{'doc_uri': 'doc_uri_1_1', 'content': 'RDDs are fundamental to Sparks fault tolerance, allowing lost data to be reconstructed.'}, {'doc_uri': 'doc_uri_1_extra', 'content': 'Lineage information in RDDs enables Spark to compute only the lost data.'}]}, {'request_id': 'f7be7cc6-264e-4cc8-83ef-b61903eb9b8c', 'app_version': 'version_name_1', 'response': 'The Spark Driver manages task scheduling and execution, acting as the control point for Spark applications.', 'retrieval_context': [{'doc_uri': 'doc_uri_2_1', 'content': 'The Driver node plays a critical role in launching tasks on executors.'}, {'doc_uri': 'doc_uri_2_extra', 'content': 'It is responsible for translating the application into a series of tasks.'}]}, {'request_id': 'bf4e11df-3085-4863-97d5-d9577fe0b387', 'app_version': 'version_name_1', 'response': 'Spark execution stages are divided by actions requiring data shuffling, optimizing parallel execution.', 'retrieval_context': [{'doc_uri': 'doc_uri_3_1', 'content': 'Stages are Sparks division of tasks that can run concurrently.'}, {'doc_uri': 'doc_uri_3_extra', 'content': 'A shuffle operation triggers the start of a new stage.'}]}, {'request_id': 'e3bdaab0-7996-485d-8ed8-e976439ca4b2', 'app_version': 'version_name_1', 'response': 'Data skew can be managed by custom partitioning or broadcasting smaller datasets.', 'retrieval_context': [{'doc_uri': 'doc_uri_4_1', 'content': 'Addressing data skew requires strategies like salting or custom partitioners.'}, {'doc_uri': 'doc_uri_4_extra', 'content': 'Broadcasting can help avoid shuffles for skewed joins.'}]}, {'request_id': 'a8f733a1-205c-4e5d-8a16-23e81ddc2505', 'app_version': 'version_name_1', 'response': 'DataFrames provide optimizations and ease of use over RDDs with their schema-aware operations.', 'retrieval_context': [{'doc_uri': 'doc_uri_5_1', 'content': 'DataFrames offer a higher-level API and optimizations through Catalyst.'}, {'doc_uri': 'doc_uri_5_extra', 'content': 'They abstract away the complexity of RDDs, making data manipulation more straightforward.'}]}], 'spark_bot_v1_answer_sheet_minimal': [{'request_id': 'd2f19cb5-8d15-4e5a-a131-800b49d7c28f', 'app_version': 'version_name_1', 'response': 'reduceByKey aggregates data before shuffling, whereas groupByKey shuffles all data, making reduceByKey more efficient.'}, {'request_id': '60ed4657-5acc-4960-a25b-c91969b28c2c', 'app_version': 'version_name_1', 'response': 'Minimizing data shuffling in Spark involves using narrow transformations and avoiding operations like groupByKey when possible.'}, {'request_id': '2e8d7aba-a1ab-4442-a404-cc334e57fee1', 'app_version': 'version_name_1', 'response': 'Broadcast variables enhance performance by keeping a read-only variable cached on each machine, rather than shipping a copy of it with tasks.'}, {'request_id': '0ce5ddf0-23aa-4295-b950-67cbf0490574', 'app_version': 'version_name_1', 'response': 'The main components of Spark’s execution model include the Driver, Executors, and the Cluster Manager.'}, {'request_id': 'e08d1d26-ff11-406f-a6be-70712e1b7347', 'app_version': 'version_name_1', 'response': 'Increasing the number of executors in Spark can be achieved by configuring spark.executor.instances in Spark’s configuration.'}, {'request_id': 'f6ed004d-5df1-4eb8-b0dc-13c37180543a', 'app_version': 'version_name_1', 'response': 'Spark ensures fault tolerance through its RDDs, which can rebuild data using lineage.'}, {'request_id': 'f7be7cc6-264e-4cc8-83ef-b61903eb9b8c', 'app_version': 'version_name_1', 'response': 'The Spark Driver manages task scheduling and execution, acting as the control point for Spark applications.'}, {'request_id': 'bf4e11df-3085-4863-97d5-d9577fe0b387', 'app_version': 'version_name_1', 'response': 'Spark execution stages are divided by actions requiring data shuffling, optimizing parallel execution.'}, {'request_id': 'e3bdaab0-7996-485d-8ed8-e976439ca4b2', 'app_version': 'version_name_1', 'response': 'Data skew can be managed by custom partitioning or broadcasting smaller datasets.'}, {'request_id': 'a8f733a1-205c-4e5d-8a16-23e81ddc2505', 'app_version': 'version_name_1', 'response': 'DataFrames provide optimizations and ease of use over RDDs with their schema-aware operations.'}], 'spark_bot_v2_answer_sheet': [{'request_id': 'f6ed004d-5df1-4eb8-b0dc-13c37180543a', 'app_version': 'version_name_2', 'response': 'In Spark, fault tolerance is achieved through lineage information, allowing for efficient data recovery.', 'retrieval_context': [{'doc_uri': 'doc_uri_1_1', 'content': 'Lineage in Spark allows for the reconstruction of lost data through its computation graph.'}, {'doc_uri': 'doc_uri_1_2', 'content': 'Fault tolerance is ensured by RDDs, which track transformations.'}]}, {'request_id': 'f7be7cc6-264e-4cc8-83ef-b61903eb9b8c', 'app_version': 'version_name_2', 'response': 'The Driver in Spark acts as the brain of the application, coordinating the execution of tasks.', 'retrieval_context': [{'doc_uri': 'doc_uri_2_1', 'content': 'The Driver program runs the applications main function and schedules tasks.'}, {'doc_uri': 'doc_uri_2_2', 'content': 'It communicates with the cluster manager and oversees task execution.'}]}, {'request_id': 'bf4e11df-3085-4863-97d5-d9577fe0b387', 'app_version': 'version_name_2', 'response': 'Sparks execution model divides work into stages, optimized by the data processing needs.', 'retrieval_context': [{'doc_uri': 'doc_uri_3_1', 'content': 'Execution stages are determined by data shuffling requirements.'}, {'doc_uri': 'doc_uri_3_2', 'content': 'Stages optimize task execution across the cluster.'}]}, {'request_id': 'e3bdaab0-7996-485d-8ed8-e976439ca4b2', 'app_version': 'version_name_2', 'response': 'Managing skew involves strategic partitioning and leveraging data broadcasting.', 'retrieval_context': [{'doc_uri': 'doc_uri_4_1', 'content': 'Skewed data can be mitigated by adjusting Sparks partitioner settings.'}, {'doc_uri': 'doc_uri_4_2', 'content': 'Broadcast variables help minimize the impact of data skew on join operations.'}]}, {'request_id': 'a8f733a1-205c-4e5d-8a16-23e81ddc2505', 'app_version': 'version_name_2', 'response': 'DataFrames enhance usability and performance with built-in optimizations.', 'retrieval_context': [{'doc_uri': 'doc_uri_5_1', 'content': 'DataFrames provide a more accessible API for data operations than RDDs.'}, {'doc_uri': 'doc_uri_5_2', 'content': 'Their optimized execution engine speeds up processing.'}]}, {'request_id': 'e08d1d26-ff11-406f-a6be-70712e1b7347', 'app_version': 'version_name_2', 'response': 'Increasing executors enhances parallelism and can be configured in Spark settings.', 'retrieval_context': [{'doc_uri': 'doc_uri_6_1', 'content': 'To scale up processing, adjust the spark.executor.instances parameter.'}, {'doc_uri': 'doc_uri_6_2', 'content': 'More executors allow for better distribution of tasks and faster computation.'}]}, {'request_id': 'd2f19cb5-8d15-4e5a-a131-800b49d7c28f', 'app_version': 'version_name_2', 'response': 'reduceByKey is more efficient due to pre-shuffle data reduction, unlike groupByKey.', 'retrieval_context': [{'doc_uri': 'doc_uri_7_1', 'content': 'reduceByKey aggregates data locally before shuffling, reducing network load.'}]}, {'request_id': '60ed4657-5acc-4960-a25b-c91969b28c2c', 'app_version': 'version_name_2', 'response': 'To minimize shuffling, use transformations that limit data movement and consider data partitioning strategies.', 'retrieval_context': [{'doc_uri': 'doc_uri_8_1', 'content': 'Optimizing transformations and actions can significantly reduce shuffle.'}, {'doc_uri': 'doc_uri_8_2', 'content': 'Careful data partitioning helps in achieving more efficient data processing.'}]}, {'request_id': '2e8d7aba-a1ab-4442-a404-cc334e57fee1', 'app_version': 'version_name_2', 'response': 'Broadcast variables are used to enhance Spark performance by caching data on all nodes.', 'retrieval_context': [{'doc_uri': 'doc_uri_9_1', 'content': 'Utilizing broadcast variables minimizes the need for data to be sent to each task.'}, {'doc_uri': 'doc_uri_9_2', 'content': 'They are crucial for performing efficient joins and data lookup operations.'}]}, {'request_id': '0ce5ddf0-23aa-4295-b950-67cbf0490574', 'app_version': 'version_name_2', 'response': 'Spark’s architecture comprises the Driver, Executors, and Cluster Manager, each playing a key role in distributed processing.', 'retrieval_context': [{'doc_uri': 'doc_uri_10_1', 'content': 'The Driver orchestrates the execution of tasks, while Executors perform them.'}, {'doc_uri': 'doc_uri_10_2', 'content': 'The Cluster Manager allocates resources across the cluster for task execution.'}]}], 'spark_bot_v2_answer_sheet_minimal': [{'request_id': 'f6ed004d-5df1-4eb8-b0dc-13c37180543a', 'app_version': 'version_name_2', 'response': 'In Spark, fault tolerance is achieved through lineage information, allowing for efficient data recovery.'}, {'request_id': 'f7be7cc6-264e-4cc8-83ef-b61903eb9b8c', 'app_version': 'version_name_2', 'response': 'The Driver in Spark acts as the brain of the application, coordinating the execution of tasks.'}, {'request_id': 'bf4e11df-3085-4863-97d5-d9577fe0b387', 'app_version': 'version_name_2', 'response': 'Sparks execution model divides work into stages, optimized by the data processing needs.'}, {'request_id': 'e3bdaab0-7996-485d-8ed8-e976439ca4b2', 'app_version': 'version_name_2', 'response': 'Managing skew involves strategic partitioning and leveraging data broadcasting.'}, {'request_id': 'a8f733a1-205c-4e5d-8a16-23e81ddc2505', 'app_version': 'version_name_2', 'response': 'DataFrames enhance usability and performance with built-in optimizations.'}, {'request_id': 'e08d1d26-ff11-406f-a6be-70712e1b7347', 'app_version': 'version_name_2', 'response': 'Increasing executors enhances parallelism and can be configured in Spark settings.'}, {'request_id': 'd2f19cb5-8d15-4e5a-a131-800b49d7c28f', 'app_version': 'version_name_2', 'response': 'reduceByKey is more efficient due to pre-shuffle data reduction, unlike groupByKey.'}, {'request_id': '60ed4657-5acc-4960-a25b-c91969b28c2c', 'app_version': 'version_name_2', 'response': 'To minimize shuffling, use transformations that limit data movement and consider data partitioning strategies.'}, {'request_id': '2e8d7aba-a1ab-4442-a404-cc334e57fee1', 'app_version': 'version_name_2', 'response': 'Broadcast variables are used to enhance Spark performance by caching data on all nodes.'}, {'request_id': '0ce5ddf0-23aa-4295-b950-67cbf0490574', 'app_version': 'version_name_2', 'response': 'Spark’s architecture comprises the Driver, Executors, and Cluster Manager, each playing a key role in distributed processing.'}], 'spark_bot_v3_answer_sheet': [{'request_id': 'f6ed004d-5df1-4eb8-b0dc-13c37180543a', 'app_version': 'version_name_3', 'response': 'In Spark, fault tolerance is not achieved through lineage information, preventing efficient data recovery.', 'retrieval_context': [{'doc_uri': 'doc_uri_1_1', 'content': 'Lineage in Spark allows for the reconstruction of lost data through its computation graph.'}, {'doc_uri': 'doc_uri_1_2', 'content': 'Fault tolerance is ensured by RDDs, which track transformations.'}]}, {'request_id': 'f7be7cc6-264e-4cc8-83ef-b61903eb9b8c', 'app_version': 'version_name_3', 'response': 'The Driver in Spark does not act as the brain of the application, failing to coordinate the execution of tasks.'}, {'request_id': 'bf4e11df-3085-4863-97d5-d9577fe0b387', 'app_version': 'version_name_3', 'response': "Spark's execution model does not divide work into stages, not optimized by the data processing needs."}, {'request_id': 'e3bdaab0-7996-485d-8ed8-e976439ca4b2', 'app_version': 'version_name_3', 'response': 'Managing skew does not involve strategic partitioning nor leveraging data broadcasting.'}, {'request_id': 'a8f733a1-205c-4e5d-8a16-23e81ddc2505', 'app_version': 'version_name_3', 'response': 'DataFrames do not enhance usability nor performance, lacking built-in optimizations.'}, {'request_id': 'e08d1d26-ff11-406f-a6be-70712e1b7347', 'app_version': 'version_name_3', 'response': 'Increasing executors does not enhance parallelism and cannot be configured in Spark settings.', 'retrieval_context': [{'doc_uri': 'doc_uri_6_1', 'content': 'To scale up processing, adjust the spark.executor.instances parameter.'}, {'doc_uri': 'doc_uri_6_2', 'content': 'More executors allow for better distribution of tasks and faster computation.'}]}, {'request_id': 'd2f19cb5-8d15-4e5a-a131-800b49d7c28f', 'app_version': 'version_name_3', 'response': 'reduceByKey is not more efficient due to pre-shuffle data reduction, just like groupByKey.'}, {'request_id': '60ed4657-5acc-4960-a25b-c91969b28c2c', 'app_version': 'version_name_3', 'response': 'To maximize shuffling, avoid transformations that limit data movement and disregard data partitioning strategies.'}, {'request_id': '2e8d7aba-a1ab-4442-a404-cc334e57fee1', 'app_version': 'version_name_3', 'response': 'Broadcast variables are not used to enhance Spark performance nor caching data on all nodes.', 'retrieval_context': [{'doc_uri': 'doc_uri_9_1'}, {'doc_uri': 'doc_uri_9_2'}]}, {'request_id': '0ce5ddf0-23aa-4295-b950-67cbf0490574', 'app_version': 'version_name_3', 'response': 'Spark’s architecture does not comprise the Driver, Executors, and Cluster Manager, each failing a key role in distributed processing.', 'retrieval_context': [{'doc_uri': 'doc_uri_10_1', 'content': 'The Driver orchestrates the execution of tasks, while Executors perform them.'}, {'doc_uri': 'doc_uri_10_2', 'content': 'The Cluster Manager allocates resources across the cluster for task execution.'}]}], 'spark_bot_v3_answer_sheet_minimal': [{'request_id': 'f6ed004d-5df1-4eb8-b0dc-13c37180543a', 'app_version': 'version_name_3', 'response': 'In Spark, fault tolerance is not achieved through lineage information, preventing efficient data recovery.'}, {'request_id': 'f7be7cc6-264e-4cc8-83ef-b61903eb9b8c', 'app_version': 'version_name_3', 'response': 'The Driver in Spark does not act as the brain of the application, failing to coordinate the execution of tasks.'}, {'request_id': 'bf4e11df-3085-4863-97d5-d9577fe0b387', 'app_version': 'version_name_3', 'response': "Spark's execution model does not divide work into stages, not optimized by the data processing needs."}, {'request_id': 'e3bdaab0-7996-485d-8ed8-e976439ca4b2', 'app_version': 'version_name_3', 'response': 'Managing skew does not involve strategic partitioning nor leveraging data broadcasting.'}, {'request_id': 'a8f733a1-205c-4e5d-8a16-23e81ddc2505', 'app_version': 'version_name_3', 'response': 'DataFrames do not enhance usability nor performance, lacking built-in optimizations.'}, {'request_id': 'e08d1d26-ff11-406f-a6be-70712e1b7347', 'app_version': 'version_name_3', 'response': 'Increasing executors does not enhance parallelism and cannot be configured in Spark settings.'}, {'request_id': 'd2f19cb5-8d15-4e5a-a131-800b49d7c28f', 'app_version': 'version_name_3', 'response': 'reduceByKey is not more efficient due to pre-shuffle data reduction, just like groupByKey.'}, {'request_id': '60ed4657-5acc-4960-a25b-c91969b28c2c', 'app_version': 'version_name_3', 'response': 'To maximize shuffling, avoid transformations that limit data movement and disregard data partitioning strategies.'}, {'request_id': '2e8d7aba-a1ab-4442-a404-cc334e57fee1', 'app_version': 'version_name_3', 'response': 'Broadcast variables are not used to enhance Spark performance nor caching data on all nodes.'}, {'request_id': '0ce5ddf0-23aa-4295-b950-67cbf0490574', 'app_version': 'version_name_3', 'response': 'Spark’s architecture does not comprise the Driver, Executors, and Cluster Manager, each failing a key role in distributed processing.'}]}



# COMMAND ----------

# DBTITLE 1,Load Sample Data to Delta Tables
# Drop if exists
DROP_IF_EXISTS = False

# Create schema if not exists
spark.sql(f"create schema if not exists {catalog}.{schema}")

# Load sample data to Delta Tables
for table in SAMPLE_DATASETS:
    table_name = f"{catalog}.{schema}.{table}"
    if DROP_IF_EXISTS and spark.catalog.tableExists(table_name):
        spark.sql(f"drop table {table_name}")
    if not spark.catalog.tableExists(table_name):
        df = spark.read.json(spark.sparkContext.parallelize(RAW_DATA[table]))
        df.write.format("delta").option("mergeSchema", "true").mode(
            "overwrite"
        ).saveAsTable(table_name)
        print(f"Loaded {table_name}")
    else:
        print(f"Previously loaded {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Overview of sample data
# MAGIC
# MAGIC **Sample data with the complete schema**
# MAGIC - `spark_bot_eval_dataset_with_ground_truth` # eval dataset that includes the ground truth for all questions and all fields filled out
# MAGIC - `spark_bot_v1_answer_sheet` # chain v1 answer sheet with all fields filled out
# MAGIC - `spark_bot_v2_answer_sheet` # chain v2 answer sheet with all fields filled out
# MAGIC - `spark_bot_v3_answer_sheet`  # chain v3 answer sheet that contains all wrong answers
# MAGIC
# MAGIC **Sample data with the minimal schema**
# MAGIC - `spark_bot_eval_dataset` # eval dataset with only questions and no ground truth
# MAGIC - `spark_bot_v1_answer_sheet_minimal`  # v1 answer sheet without retrieval_context
# MAGIC - `spark_bot_v2_answer_sheet_minimal`"  # v2 answer sheet without retrieval_context
# MAGIC - `spark_bot_v3_answer_sheet_minimal`"  # v3 minimal answer sheet that contains all wrong answers
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Sample workflow: Evaluate v1 of the bot

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start by evaluating the v1 of our chain using the default evaluation configuration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Set
# MAGIC The evaluation set represents the human-annotated ground truth data.
# MAGIC
# MAGIC | Column Name                  | Type                                              | Required? | Comment                                                                                                                                                  |
# MAGIC |------------------------------|---------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC | request_id                   | STRING                                            | ✅        | Id of the request (question)                                                                                                                             |
# MAGIC | request                     | STRING                                            |           | A request (question) to the RAG app, e.g., “What is Spark?”                                                                                              |
# MAGIC | expected_response            | STRING                                            |           | (Optional) The expected answer to this question                                                                                                          |
# MAGIC | expected_retrieval_context   | ARRAY<STRUCT<doc_uri: STRING, content: STRING>>   |           | (Optional) The expected retrieval context. The entries are ordered in descending rank. Each entry can record the URI of the retrieved doc and optionally the (sub)content that was retrieved. |
# MAGIC

# COMMAND ----------

spark_bot_eval_dataset_with_ground_truth_table = f"{catalog}.{schema}.spark_bot_eval_dataset_with_ground_truth"
display(spark.table(spark_bot_eval_dataset_with_ground_truth_table))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Answer sheet
# MAGIC
# MAGIC The answer sheet includes the outputs of chain v1 - this data was created by calling the chain for each `request` in the evaluation set.
# MAGIC
# MAGIC | Column Name       | Type                                             | Required? | Comment                                                                                                          |
# MAGIC |-------------------|--------------------------------------------------|-----------|------------------------------------------------------------------------------------------------------------------|
# MAGIC | request_id        | STRING                                           | ✅        | The identifier for the request. This should match the identifier in the eval set.                                |
# MAGIC | app_version       | STRING                                           | ✅        | The version of the RAG app that was used to generate answers.                                                    |
# MAGIC | response          | STRING                                           | ✅        | The output of the RAG app on the given question                                                                  |
# MAGIC | retrieval_context | ARRAY<STRUCT<doc_uri: STRING, content: STRING>> |           | (Optional) The retrieved context which was used in the generation of the answer. This has the same structure as the expected_retrieval_context in the eval set. |
# MAGIC

# COMMAND ----------

spark_bot_v1_answer_sheet_table = f"{catalog}.{schema}.spark_bot_v1_answer_sheet"
display(spark.table(spark_bot_v1_answer_sheet_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation
# MAGIC
# MAGIC We call `evaluate(...)` pointing to the above 2 tables which will:
# MAGIC - Create an MLflow run with metadata about the evaluation configuration
# MAGIC - Create a dashboard with the metric comptuations
# MAGIC - Create two output Delta Tables
# MAGIC   - `{eval_table_name}_assessments` the LLM judge outputs for each `request`
# MAGIC   - `{eval_table_name}_eval_metrics` the computed metrics for each `request`
# MAGIC
# MAGIC **IMPORTANT: If you do not pass an evaluation configuration, Evaluation Suite will use the default configuration (all judges enabled).  However, evaluation is currently slow with the Databricks provided LLM judge due to a limitation we are working to remove.  You can temporarily use any Model Serving endpoint to overcome this limitation, including DBRX, but this requires you to pass a configuration as shown below**
# MAGIC

# COMMAND ----------

import yaml

############
# Default evaluation configuration
############
config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval",
            "assessments": [
                "harmful",
                "faithful_to_context",
                "relevant_to_question_and_context",
                "relevant_to_question",
                "answer_good",
                "context_relevant_to_question",
            ],
        }
    ]
}


############
# Currently, evaluation is slow with the Databricks provided LLM judge due to a limitation we are working to remove.  You can temporarily use any Model Serving endpoint to overcome this limitation, including DBRX.
############
config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval_dbrx",
            "endpoint_name": "endpoints:/databricks-dbrx-instruct",
            "assessments": [
                "harmful",
                "faithful_to_context",
                "relevant_to_question_and_context",
                "relevant_to_question",
                "answer_good",
                "context_relevant_to_question",
            ],
        }
    ]
}

config_yml = yaml.dump(config_json)
config_yml

# COMMAND ----------

eval_results = rag_eval.evaluate(
    eval_set_table_name=spark_bot_eval_dataset_with_ground_truth_table,
    answer_sheet_table_name=spark_bot_v1_answer_sheet_table,
    config=config_yml
)

############
# Experimental: Log evaluation results to MLflow.  Note you can also use the dashboard produced by RAG Studio to view metrics/debug quality - it has more advanced functionality.
# Known issues: Can only be run once per run_id.
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############
experimental_add_metrics_to_run(eval_results, eval_results.mlflow_run_id)
experimental_add_eval_outputs_to_run(eval_results, eval_results.mlflow_run_id)
experimental_add_eval_config_tags_to_run(eval_results, config_yml, eval_results.mlflow_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect outputs
# MAGIC
# MAGIC ### MLflow run
# MAGIC The MLflow run contains metadata about the evaluation run, including its configuration.

# COMMAND ----------

eval_results.mlflow_run_url

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dashboard
# MAGIC
# MAGIC The dashboard shows the aggregated metrics.

# COMMAND ----------

eval_results.dashboard_url

# COMMAND ----------

# MAGIC %md
# MAGIC ### Metrics
# MAGIC
# MAGIC To see the details of metrics for each `request` that are aggregated in the dashboard, you can inspect the below table.
# MAGIC
# MAGIC
# MAGIC | Metric Type | Metric Name                           | Explanation                                                             | Required Evaluation Set Fields | Required Answer Sheet Fields | 
# MAGIC |-------------|---------------------------------------|-------------------------------------------------------------------------|-----------------------|-----------------------|
# MAGIC | Response    | token_count                           | Number of tokens in the generated answer                                |    n/a                   |      response  | 
# MAGIC | Response    | llm_judged_harmful                    | Is the answer harmful? (Uses LLM-as-a-judge)                            |         request              |  response|
# MAGIC | Response    | llm_judged_faithful_to_context        | Is the answer faithful to the retrieval context? (Uses LLM-as-a-judge)  |       request                | response, retrieval_context|
# MAGIC | Response    | llm_judged_relevant_to_question_and_context | Is the answer relevant given the question and the retrieved context? (Uses LLM-as-a-judge) | request | response, retrieval_context |
# MAGIC | Response    | llm_judged_answer_good                | Is the answer good given the question and the ground-truth answer? (Uses LLM-as-a-judge) | request, expected_response |response |
# MAGIC | Retrieval   | ground_truth_precision_at_k | Precision@k (k=1,3,5, 10) for the retrieved context                     | expected_retrieval_context | retrieval_context|
# MAGIC | Retrieval   | ground_truth_recall_at_k    | Recall@k (k=1,3,5, 10) for the retrieved context                        | expected_retrieval_context | retrieval_context|
# MAGIC | Retrieval   | ground_truth_ndcg_at_k      | NDCG@k (k=1,3,5, 10) for the retrieved context                          | expected_retrieval_context | retrieval_context|
# MAGIC | Retrieval   | llm_judged_precision_at_k | Precision@k (k=1,3,5, 10) for the retrieved context                     | n/a | retrieval_context|
# MAGIC | Retrieval   | llm_judged_recall_at_k    | Recall@k (k=1,3,5, 10) for the retrieved context                        | n/a | retrieval_context|
# MAGIC | Retrieval   | llm_judged_ndcg_at_k      | NDCG@k (k=1,3,5, 10) for the retrieved context                          | n/a | retrieval_context|

# COMMAND ----------

display(spark.table(eval_results.eval_metrics_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assessments
# MAGIC
# MAGIC In order to gain confidence the judges are working as intended, you can inspect the rationale for each judge's rating of a `request`.

# COMMAND ----------

display(spark.table(eval_results.assessments_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC # Sample workflow: Evaluate v2 of the bot
# MAGIC
# MAGIC Based on the evaluation of v1, you have made changes, and are ready to evaluate the v2.  Evaluation results are tracked per evaluation set - so the evaluation results for v2 are appended to the tables created above for v1.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Answer sheet v2

# COMMAND ----------

spark_bot_v2_answer_sheet_table = f"{catalog}.{schema}.spark_bot_v2_answer_sheet"
display(spark.table(spark_bot_v2_answer_sheet_table))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation

# COMMAND ----------

eval_results = rag_eval.evaluate(
    eval_set_table_name=spark_bot_eval_dataset_with_ground_truth_table,
    answer_sheet_table_name=spark_bot_v2_answer_sheet_table,
    config=config_yml
)

############
# Experimental: Log evaluation results to MLflow.  Note you can also use the dashboard produced by RAG Studio to view metrics/debug quality - it has more advanced functionality.
# Known issues: Can only be run once per run_id.
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############
experimental_add_metrics_to_run(eval_results, eval_results.mlflow_run_id)
experimental_add_eval_outputs_to_run(eval_results, eval_results.mlflow_run_id)
experimental_add_eval_config_tags_to_run(eval_results, config_yml, eval_results.mlflow_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect outputs
# MAGIC
# MAGIC Notice that the results from v2 are APPENDED to the tables from the v1 evaluation.

# COMMAND ----------

eval_results.dashboard_url

# COMMAND ----------

eval_results.mlflow_run_url

# COMMAND ----------

display(spark.table(eval_results.eval_metrics_table_name))

# COMMAND ----------

display(spark.table(eval_results.assessments_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC # Improving the relevance of judges
# MAGIC
# MAGIC Now that we have run the evaluation for v1 and v2, let's customize the evaluation configuration with examples from your use case to improve the accuracy of the Databricks judges.  Databricks strongly reccomends providing at least 2 postive and 2 negative examples per judge to improve the accuracy.

# COMMAND ----------

# Example config where we will supply a positive and a negative example for several assesments.
# Note all assessments accept examples, but you can mix and match which assessments you provide examples for. 

config_json = {
    "assessment_judges": [
        {
            # "judge_name": "databricks_eval",
            "judge_name": "databricks_eval_dbrx",
            "endpoint_name": "endpoints:/databricks-dbrx-instruct",
            "assessments": [
                "harmful",
                "relevant_to_question_and_context",
                "relevant_to_question",
                {
                    "answer_good": {
                        "examples": [
                            {
                                "request": "What is Apache Spark?",
                                "response": "Spark is what happens when there is fire.",
                                "expected_response": "Spark is a distributed data processing engine.",
                                "value": False,
                                "rationale": "The output is completely incorrect",
                            },
                            {
                                "request": "What is RAG?",
                                "response": "Retrieval-Augmented-Generation is a powerful paradigm for using LLMs",
                                "expected_response": "RAG is retrieval augmented generation",
                                "value": True,
                                "rationale": "The output matches well the expected response.",
                            },
                        ]
                    }
                },
                {
                    "faithful_to_context": {
                        "examples": [
                            {
                                "request": "What is Apache Spark?",
                                "response": "Spark is what happens when there is fire.",
                                "context": "Apache Spark is an open-source unified analytics engine for large-scale data processing.",
                                "value": False,
                            }
                        ]
                    },
                },
                {
                    "context_relevant_to_question": {
                        "examples": [
                            {
                                "request": "What is Apache Spark?",
                                "context": "Apache Spark is an open-source unified analytics engine for large-scale data processing.",
                                "value": True,
                                "rationale": "The retrieved co\tntext accurately answers the question.",
                            }
                        ]
                    }
                },
            ],
        }
    ]
}

config_custom_judge_yml = yaml.dump(config_json)

# COMMAND ----------

eval_results = rag_eval.evaluate(
    eval_set_table_name=spark_bot_eval_dataset_with_ground_truth_table,
    answer_sheet_table_name=spark_bot_v2_answer_sheet_table,
    config=config_custom_judge_yml
)

############
# Experimental: Log evaluation results to MLflow.  Note you can also use the dashboard produced by RAG Studio to view metrics/debug quality - it has more advanced functionality.
# Known issues: Can only be run once per run_id.
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############
experimental_add_metrics_to_run(eval_results, eval_results.mlflow_run_id)
experimental_add_eval_outputs_to_run(eval_results, eval_results.mlflow_run_id)
experimental_add_eval_config_tags_to_run(eval_results, config_yml, eval_results.mlflow_run_id)
