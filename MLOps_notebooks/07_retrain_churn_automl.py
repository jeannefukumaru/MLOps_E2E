# Databricks notebook source
# MAGIC %md
# MAGIC ## Monthly AutoML Retrain
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step7.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Features

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %run ./99_retraining_utils

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

# Set config for database name, file paths, and table names
feature_table = f'{database_name}.churn_features'

fs = FeatureStoreClient()

features = fs.read_table(feature_table)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Train a new model based on latest features

# COMMAND ----------

import databricks.automl
model = databricks.automl.classify(features, 
                                   target_col = "churn",
                                   data_dir= f"dbfs:{get_default_path()}/automl",
                                   timeout_minutes=5) 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Register new model to Model Registry if it is better. Also deregisters and archives current production model if its being replaced
# MAGIC 
# MAGIC #### Additionally, also saves retraining timestamp and model information into separate Delta table for audit purposes
# MAGIC 
# MAGIC * see 99_retraining_utils.py file for more information on the `ModelRegistration` class that handles the above logic

# COMMAND ----------

# Uses Model Registry to register new model if it is better, deregisters current production model if its being replaced
model_registration = ModelRegistration(experiment_name, experiment_title, model_name, metric, direction)
model_registration.register_best(registration_message, logging_location, log_db, log_table)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### View the Delta table containing our retraining history

# COMMAND ----------

from pyspark.sql import functions as F
REGISTRY_TABLE = "airbnb.registry_status"
display(spark.table(REGISTRY_TABLE).orderBy(F.col("training_time"))
