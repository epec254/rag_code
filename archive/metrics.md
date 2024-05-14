# Metrics & LLM Judges overview
Databricks provides a set of metrics that enable you to measure the quality, cost and latency of your RAG app. These metrics are curated by Databricks' Research team as the most relevant (no pun intended) metrics for evaluating RAG applications.

RAG metrics can be computed using either:
1. Human-labeled ground truth assessments
2. LLM judge-labeled assessments 

A subset of the metrics work only with *either* LLM judge-labeled OR human-labeled ground truth asessments.
## Unstructured docs retrieval & generation metrics

### Retriever

RAG Studio supports the following metrics for evaluating the retriever.

| Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments & judge name | 
|-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
| Are the retrieved chunks relevant to the user’s query?                            | Precision of "relevant chunk" @ K | 0 to 100% | 0 to 100% | ✔️ | ✔️ `context_relevant_to_question` |
| Are **ALL** chunks that are relevant to the user’s query retrieved?               | Recall of "relevant chunk" @ K | 0 to 100% |0 to 100% | ✔️ |✖️ |
| Are the retrieved chunks returned in the correct order of most to least relevant? | nDCG of "relevant chunk" @ K | 0 to 1 | 0 to 1 |✔️ | ✖️ |

### Generation model

These metrics measure the generation model's performance when the prompt is augemented with unstructured docs from a retrieval step.

| Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments & judge name | 
|-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
| Is the LLM not hallucinating & responding based ONLY on the context provided? | Faithfulness (to context) | true/false | 0 to 100% | ✖️ | ✔️ `faithful_to_context` |
| Is the response on-topic given the query AND retrieved contexts? | Answer relevance (to query given the context) | true/false | 0 to 100% | ✖️ | ✔️ `relevant_to_question_and_context` | 
| Is the response on-topic given the query? | Answer relevance (to query) | true/false | 0 to 100% | ✖️ | ✔️ `relevant_to_question` | 
| What is the cost of the generation? | Token Count | sum(tokens) | sum(tokens) | n/a |n/a |
| What is the latency of generation? | Latency | milliseconds | average(milliseconds) | n/a | n/a |

### RAG chain metrics

These metrics measure the chain's final response back to the user.  

| Question to answer                                                                | Metric | Per trace value | Aggregated value | Work with human assessments | LLM judged assessments & judge name | 
|-----------------------------------------------------------------------------------|--------|--------|--------|------|--------|
| Is the response accurate (correct)? | Answer correctness (vs. ground truth) | true/false | 0 to 100% |✔️ `answer_good` | ✖️ |
| Does the response violate any of my company policies (racism, toxicity, etc)? | Toxicity | true/false | 0 to 100% | ✖️ | ✔️ `harmful` |