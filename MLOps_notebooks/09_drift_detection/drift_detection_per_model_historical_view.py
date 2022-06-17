# Databricks notebook source
# MAGIC %pip install evidently

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import json
import pandas as pd
import numpy as np
import requests
import zipfile
import io

import plotly.offline as py #working offline
import plotly.graph_objs as go

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# COMMAND ----------

target = "price"
numerical_features = ["accommodates", "bedrooms", "beds", "minimum_nights", "number_of_reviews", "number_of_reviews_ltm", "review_scores_rating",]
categorical_features = ["host_is_superhost", "neighbourhood_cleansed", "property_type", "room_type"]

data_columns = ColumnMapping()
data_columns.target = target
data_columns.numerical_features = numerical_features
data_columns.categorical_features = categorical_features

# COMMAND ----------

#set reference dates
reference_date = parser.parse('2022-06-03').date()

#set experiment batches dates
experiment_batches = [
   parser.parse('2022-07-03').date(),
   parser.parse('2022-08-03').date(),
]

raw_data = spark.table("airbnb.airbnb_hawaii").toPandas()

# COMMAND ----------

raw_data["monitoring_date"].unique()

# COMMAND ----------

#evaluate data drift with Evidently Profile
def detect_dataset_drift(reference, production, column_mapping, get_ratio=False):
    """
    Returns True if Data Drift is detected, else returns False.
    If get_ratio is True, returns ration of drifted features.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    Data Drift for the dataset is detected if share of the drifted features is above the selected threshold (default value is 0.5).
    """
    
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)
        
    n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
    n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]
    
    if get_ratio:
        return n_drifted_features / n_features
    else:
        return json_report["data_drift"]["data"]["metrics"]["dataset_drift"]

# COMMAND ----------

#evaluate data drift with Evidently Profile
def detect_features_drift(reference, production, column_mapping, get_scores=False):
    """
    Returns 1 if Data Drift is detected, else returns 0. 
    If get_scores is True, returns scores value (like p-value) for each feature.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    """
    
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)
    
    drifts = []
    num_features = column_mapping.numerical_features if column_mapping.numerical_features else []
    cat_features = column_mapping.categorical_features if column_mapping.categorical_features else []
    for feature in num_features + cat_features:
        drift_score = json_report['data_drift']['data']['metrics'][feature]['drift_score']
        if get_scores:
            drifts.append((feature, drift_score))
        else:
            drifts.append((feature, json_report['data_drift']['data']['metrics'][feature]['drift_detected']))
             
    return drifts

# COMMAND ----------

raw_data[raw_data["monitoring_date"] == experiment_batches[0]]

# COMMAND ----------

features_historical_drift = []

for date in experiment_batches:
    drifts = detect_features_drift(raw_data[raw_data["monitoring_date"] == reference_date],
                           raw_data[raw_data["monitoring_date"] == date], 
                           column_mapping=data_columns)
    print(drifts)
    features_historical_drift.append([x[1] for x in drifts])
    
features_historical_drift_frame = pd.DataFrame(features_historical_drift, 
                                               columns = data_columns.numerical_features + data_columns.categorical_features)

# COMMAND ----------

fig = go.Figure(data=go.Heatmap(
                   z = features_historical_drift_frame.astype(int).transpose(),
                   x = [x for x in experiment_batches],
                   y = data_columns.numerical_features,
                   hoverongaps = False,
                   xgap = 1,
                   ygap = 1,
                   zmin = 0,
                   zmax = 1,
                   showscale = False,
                   colorscale = 'Bluered'
))

fig.update_xaxes(side="top")

fig.update_layout(
    xaxis_title = "Timestamp",
    yaxis_title = "Feature Drift"
)

fig.show()

# COMMAND ----------

features_historical_drift_pvalues = []

for batch in experiment_batches:
    drifts = detect_features_drift(raw_data[raw_data["date"]=="2022-01-01"], 
                           raw_data[raw_data["date"]==batch], 
                           column_mapping=data_columns,
                           get_scores=True)
    
    features_historical_drift_pvalues.append([x[1] for x in drifts])
    
features_historical_drift_pvalues_frame = pd.DataFrame(features_historical_drift_pvalues, 
                                                       columns = data_columns.numerical_features + data_columns.categorical_features)

# COMMAND ----------

fig = go.Figure(data=go.Heatmap(
                   z = features_historical_drift_pvalues_frame.transpose(),
                   x = [x for x in experiment_batches],
                   y = features_historical_drift_pvalues_frame.columns,
                   hoverongaps = False,
                   xgap = 1,
                   ygap = 1,
                   zmin = 0,
                   zmax = 1,
                   colorscale = 'reds_r'
                   )
               )

fig.update_xaxes(side="top")

fig.update_layout(
    xaxis_title = "Timestamp",
    yaxis_title = "p-value"
)

fig.show()

# COMMAND ----------

dataset_historical_drift = []

for batch in experiment_batches:
    dataset_historical_drift.append(detect_dataset_drift(raw_data[raw_data["date"]=="2022-01-01"], 
                           raw_data[raw_data["date"]==batch], 
                           column_mapping=data_columns))

# COMMAND ----------

fig = go.Figure(data=go.Heatmap(
                   z = [[1 if x == True else 0 for x in dataset_historical_drift]],
                   x = [x for x in experiment_batches],
                   y = [''],
                   hoverongaps = False,
                   xgap = 1,
                   ygap = 1,
                   zmin = 0,
                   zmax = 1,
                   colorscale = 'Bluered',
                   showscale = False
                   )
               )

fig.update_xaxes(side="top")

fig.update_layout(
    xaxis_title = "Timestamp",
    yaxis_title = "Dataset Drift"
)
fig.show()

# COMMAND ----------

dataset_historical_drift_ratio = []

for date in experiment_batches:
    dataset_historical_drift_ratio.append(detect_dataset_drift(raw_data[raw_data["date"]=="2022-01-01"], 
                           raw_data[raw_data["date"]==batch],
                           column_mapping=data_columns,
                           get_ratio=True))

# COMMAND ----------

fig = go.Figure(data=go.Heatmap(
                   z = [dataset_historical_drift_ratio],
                   x = [x for x in experiment_batches],
                   y = [''],
                   hoverongaps = False,
                   xgap = 1,
                   ygap = 1,
                   zmin = 0.5,
                   zmax = 1,
                   colorscale = 'reds'
                  )
               )

fig.update_xaxes(side="top")

fig.update_layout(
    xaxis_title = "Timestamp",
    yaxis_title = "Dataset Drift"
)
fig.show()

# COMMAND ----------


