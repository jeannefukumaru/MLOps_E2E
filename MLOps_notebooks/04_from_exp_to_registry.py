# Databricks notebook source
# MAGIC %md
# MAGIC ### Managing the model lifecycle with Model Registry
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step4.png?raw=true">
# MAGIC 
# MAGIC One of the primary challenges among data scientists and ML engineers is the absence of a central repository for models, their versions, and the means to manage them throughout their lifecycle.  
# MAGIC 
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) addresses this challenge and enables members of the data team to:
# MAGIC <br><br>
# MAGIC * **Discover** registered models, current stage in model development, experiment runs, and associated code with a registered model
# MAGIC * **Transition** models to different stages of their lifecycle
# MAGIC * **Deploy** different versions of a registered model in different stages, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * **Test** models in an automated fashion
# MAGIC * **Document** models throughout their lifecycle
# MAGIC * **Secure** access and permission for model registrations, transitions or modifications
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg">

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use the Model Registry
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters. During the course of this development cycle, they will select the best run within an experiment and register its model with the registry.  Think of this as **committing** the model to the registry, much as you would commit code to a version control system.  
# MAGIC 
# MAGIC The registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`. Each stage has a unique meaning. For example, `Staging` is meant for model testing, while `Production` is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# MAGIC %run ./Shared_Include

# COMMAND ----------

# MAGIC %md
# MAGIC In notebook `02_automl_baseline` we have registered our model to Mlflow Model Registry.
# MAGIC At this point the model will be in `None` stage.  Let's update the description before moving it to `Staging`.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Update Description

# COMMAND ----------

dbutils.widgets.text('model_name', defaultValue='churn_model_name')
dbutils.widgets.text('model_version', defaultValue='1')
dbutils.widgets.text('model_description', defaultValue='"This model predicts whether a customer will churn using features from the ibm_telco_churn database.  It is used to update the Telco Churn Dashboard in SQL Analytics."')
dbutils.widgets.text('comment', defaultValue="This was the best model from AutoML, I think we can use it as a baseline.")

# COMMAND ----------

churn_model_name = dbutils.widgets.get('model_name')
model_version = dbutils.widgets.get('model_version')

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_details = client.get_model_version(name=churn_model_name, version=model_version)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

description = dbutils.widgets.get("model_description")

# update model with description
client.update_registered_model(
  name=model_details.name,
  description=description
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Request Transition to Staging
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/webhooks2.png?raw=true" width = 800>

# COMMAND ----------

import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
  return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()

# COMMAND ----------

# Transition request to staging
staging_request = {'name': model_details.name,
                   'version': model_details.version,
                   'stage': 'Staging',
                   'archive_existing_versions': 'true'}

mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))

# COMMAND ----------

staging_request

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests
comment = dbutils.widgets.get("comment")
comment_body = {'name': model_details.name, 'version': model_details.version, 'comment': comment}
mlflow_call_endpoint('comments/create', 'POST', json.dumps(comment_body))

# COMMAND ----------


