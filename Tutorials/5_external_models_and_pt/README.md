# Tutorial: Using External Models or Provisioned Throughput 

1. Create an [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) or [Provisioned Throughput](https://docs.databricks.com/en/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html#) endpoint
    - See [1_openai_as_external_model](1_openai_as_external_model.py) for a click-to-run Notebook to create External Models for OpenAI and Azure OpenAI endpoints

2. Once your endpoint is created, update your RAG `chain.py` to reference the model using [`ChatDatabricks`](https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.databricks.ChatDatabricks.html) (for chat models) or [`Databricks`](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.databricks.Databricks.html#langchain_community.llms.databricks.Databricks) (for completions models).

```
from langchain_community.chat_models import ChatDatabricks

model = ChatDatabricks(
    endpoint="name-of-your-external-model-or-pt-endpoint",
    # extra_params={}, # optional e.g., temperature, etc
)

chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
    }
    | prompt
    | model
    | StrOutputParser()
)
```

3. When logging your chain, RAG Studio will automaticaly detect the endpoint and provision the appropiate credentials for your deployed model to use it.