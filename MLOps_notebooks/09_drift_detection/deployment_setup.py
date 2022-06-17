# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Run setup and utils notebooks

# COMMAND ----------

# MAGIC %run ./data_and_training_setup

# COMMAND ----------

# MAGIC %run ./monitoring_utils

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Month 0
# MAGIC 
# MAGIC * Train an inital model to predict listing prices and deploy to Production

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### i. Initial Data load
# MAGIC 
# MAGIC Load the first month of data which we use to train and evaluate our first model. 
# MAGIC 
# MAGIC We create a "Gold" table to which we will be appending each subsequent month of data.

# COMMAND ----------

delta_path = data_project_dir + "monthly_data_delta"

# COMMAND ----------

# MAGIC %md
# MAGIC #### ii. Model Training

# COMMAND ----------

# Set the month number - used for naming the MLflow run and tracked as a parameter 
month = "0"

# Specify name of MLflow run
run_name = f"month_{month}"

# Define the parameters to pass in the RandomForestRegressor model
model_params = {"n_estimators": 500,
                "max_depth": 5,
                "max_features": "log2"}

# Define a dictionary of parameters that we would like to use during preprocessing
misc_params = {"month": month,
               "target_col": target_col,
               "cat_cols": cat_cols,
               "num_cols": num_cols}

# Trigger model training and logging to MLflow
month_0_run = train_sklearn_rf_model(run_name, 
                                     delta_path, 
                                     model_params, 
                                     misc_params)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### iii. Model Deployment
# MAGIC 
# MAGIC We first register the model to the [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html). For demonstration purposes we will immediately transition the model to the "Production" stage in the MLflow Model Registry, however in a real world scenario one should have a robust model validation process in place prior to migrating a model to Production. 
# MAGIC 
# MAGIC We will demonstrate a multi-stage approach in the subsequent sections, first transitioning a model to "Staging", conducting model validation checks, and only then triggering a transition from Staging to Production once these checks are satistified.

# COMMAND ----------

# Register model to MLflow Model Registry
month_0_run_id = month_0_run.info.run_id
month_0_model_version = mlflow.register_model(model_uri=f"runs:/{month_0_run_id}/model", name=registry_model_name)

# COMMAND ----------

# Transition model to Production
month_0_model_version = transition_model(month_0_model_version, stage="Production")
print(month_0_model_version)
