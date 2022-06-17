# Databricks notebook source
# MAGIC %run ./model_monitor

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure model monitor and register baseline dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE airbnb.airbnb_hawaii

# COMMAND ----------

monitor = ModMon("airbnb", "airbnb_hawaii")
table_name = monitor.register_baseline_dataset()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate data coming in from June, July and August
# MAGIC 
# MAGIC We want to monitor the model monthly to detect:  
# MAGIC 1. Feature drift
# MAGIC 2. Label drift

# COMMAND ----------

month_1_df = spark.table("airbnb.airbnb_monthly_input_data").where(F.col("month")=="1")
month_1_df.withColumn("monitoring_date", F.to_date(F.lit("2022-07-03"))).write.format("delta").mode("append").saveAsTable(table_name)

# COMMAND ----------

month_2_df = spark.table("airbnb.airbnb_monthly_input_data").where(F.col("month")=="2")
month_2_df.withColumn("monitoring_date", F.to_date(F.lit("2022-08-03"))).write.format("delta").mode("append").saveAsTable(table_name)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Get reference and production datasets for comparison

# COMMAND ----------

ref_start_end = ("2022-06-01", "2022-06-30")
prod_start_end = ("2022-07-01", "2022-07-31")
ref_df, prod_df = monitor.get_ref_prod_datasets(table_name, ref_start_end, prod_start_end)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define column mappings so Evidently can generate drift reports

# COMMAND ----------

target = "price"
numerical_features = ["accommodates", "bedrooms", "beds", "minimum_nights", "number_of_reviews", "number_of_reviews_ltm", "review_scores_rating"]
categorical_features = ["host_is_superhost", "neighbourhood_cleansed", "property_type", "room_type"]

colmap = ColumnMapping()
colmap.target = target
colmap.numerical_features = numerical_features
colmap.categorical_features = categorical_features

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Analyze label drift

# COMMAND ----------

ld = ModMonLabelDrift(ref_df, prod_df)
dashboard = ld.report_label_drift(NumTargetDriftTab, colmap)
dashboard.show(mode="inline")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save label drift data to a table

# COMMAND ----------

# ld.save_label_drift_data(NumTargetDriftProfileSection, colmap, "airbnb", "airbnb_hawaii")

# COMMAND ----------

display(spark.table("airbnb.airbnb_hawaii")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Analyze Dataset Drift

# COMMAND ----------

ld = ModMonDatasetDrift(ref_df, prod_df)
dashboard = ld.report_dataset_drift(colmap)
dashboard.show(mode="inline")
# ld.save_dataset_drift_data(colmap, "airbnb", "airbnb_hawaii")

# COMMAND ----------

# add data quality
