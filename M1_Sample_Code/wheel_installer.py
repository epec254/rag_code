# Databricks notebook source
# MAGIC %pip install "PUT_RAG_STUDIO_WHEEL_HERE"
# MAGIC %pip install "PUT_RAG_EVAL_SUITE_WHEEL_HERE"
# MAGIC
# MAGIC ## Installing Tracing
# MAGIC %pip install opentelemetry-api opentelemetry-sdk databricks-vectorsearch tiktoken langchain langchainhub faiss-cpu -U -q
# MAGIC %pip uninstall mlflow mlflow-skinny -y # uninstall existing mlflow to avoid installation issues
# MAGIC
# MAGIC %pip install "PUT_MLFLOW_WHEEL_HERE" -U
# MAGIC %pip install "PUT_MLFLOW_SKINNY_WHEEL_HERE" -U
