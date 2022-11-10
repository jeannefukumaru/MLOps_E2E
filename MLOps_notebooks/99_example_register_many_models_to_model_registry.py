# Databricks notebook source
import mlflow

# COMMAND ----------

exp_name = "/databricks_automl/churnString_bronze_customers-2022_06_01-10_20"

# COMMAND ----------

e_id = mlflow.get_experiment_by_name(exp_name).experiment_id

# COMMAND ----------

runs = mlflow.search_runs(experiment_ids=[e_id], order_by=["metrics.test_score DESC"]).iloc[:50, :]

# COMMAND ----------

runs.dtypes

# COMMAND ----------

import time
start = time.time()
model_id = 0
for r in runs['run_id']: 
    mlflow.register_model(model_uri=f"runs:/{r}/model", name=f"churn_model_{model_id}")
    model_id += 1
    print('model registered')
end = time.time()
diff=end - start
print(diff)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
from typing import Iterator
import pandas as pd

schema = StructType([StructField('model_id', StringType(), True)])

# @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def func(runs_df: pd.Series) -> pd.Series:
    model_id = 0
    registered_models = []
    for r in runs_df['run_id']:
        model_uri = f"runs:/{r}/model"
        model_name = f"churn_model_{model_id}"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        model_id += 1
        registered_models.append({"model_uri": model_uri, "model_name": model_name})
    return pd.DataFrame(registered_models)
        
spark_runs = spark.createDataFrame(runs[['run_id', 'experiment_id']]).groupby('run_id').applyInPandas(func, schema='run_id string, experiment_id string')

# COMMAND ----------

spark_runs.show()

# COMMAND ----------


