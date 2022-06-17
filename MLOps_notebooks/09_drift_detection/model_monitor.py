# Databricks notebook source
# MAGIC %pip install evidently

# COMMAND ----------

# MAGIC %run ./data_and_training_setup

# COMMAND ----------

# MAGIC %run ./deployment_setup

# COMMAND ----------

import json as js
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import (
    DataDriftTab,
    CatTargetDriftTab,
    NumTargetDriftTab
)

from evidently.model_profile import Profile
from evidently.model_profile.sections import (
    DataDriftProfileSection,
    CatTargetDriftProfileSection,
    NumTargetDriftProfileSection
)
from evidently.pipeline.column_mapping import ColumnMapping

import json
import pandas as pd
import numpy as np
import plotly.offline as py #working offline
import plotly.graph_objs as go

# COMMAND ----------

from pyspark.sql import functions as F

class ModMon():
    def __init__(self, project_name, model_name):
        self.project_name = project_name
        self.model_name = model_name
    
    def register_baseline_dataset(self, date_colname="monitoring_date"):
        """
        log data used to train the model as a comparison baseline
        """
        run = get_run_from_registered_model(registry_model_name=self.model_name)
        baseline_df = load_delta_table_from_run(run)
        
        if date_colname in baseline_df.schema.names:  # add timestamp col for later analysis
            return print("monitoring timestamp colname clashes with existing colname, please choose another one")
        
        else:
            baseline_df = baseline_df.withColumn(date_colname, F.current_date())
            
            create_db = f"CREATE DATABASE IF NOT EXISTS {self.project_name}"
            spark.sql(create_db)
            print(f"using database {self.project_name}")
        
            table_name = f"{self.project_name}.{self.model_name}"
            drop_table = f"DROP TABLE IF EXISTS {table_name}"
            spark.sql(drop_table)
            baseline_df.write.mode("overwrite").format("delta").saveAsTable(table_name)
            print(f"baseline data saved to {table_name}")
            return table_name
    
    def get_ref_prod_datasets(self, table_name, ref_start_end, prod_start_end, sample_fraction=1.0, date_colname="monitoring_date"):
        """
        collate reference and production datasets for drift comparison
        ref_start_end: tuple of start and end dates used to slice reference data
        prod_start_end: tuple of start and end dates used to slice incoming data 
        returns tuple of ref and prod datasets as pandas df
        """
        monitor_df = spark.table(table_name)
        ref_df = monitor_df.where((F.col(date_colname) > ref_start_end[0]) & (F.col(date_colname) < ref_start_end[1])).toPandas()
        prod_df = monitor_df.where((F.col(date_colname) > prod_start_end[0]) & (F.col(date_colname) < prod_start_end[1])).toPandas()
        return ref_df, prod_df

# COMMAND ----------

class ModMonLabelDrift:
    def __init__(self, ref, prod):
        self.ref = ref
        self.prod = prod
        
    def report_label_drift(self, target_drift_type, colmap, html_filepath=None, save_html=False):
        """
        create label drift report
        
        Param
        -----
        target_drift_type: either NumTargetDriftTab or CatTargetDriftTab
        colmap: ColumnMapping saving column type information
        html_filepath: path to save report as html
        save_html: choose to save report as html
        """
        dashboard = Dashboard(tabs=[target_drift_type(verbose_level=1)])
        dashboard.calculate(self.ref, self.prod, column_mapping=colmap)
        if save_html == True:
            dashboard.save(html_filepath)
        return dashboard

    def save_label_drift_data(self, target_profile_section, colmap, project_name, model_name):
        """
        save label drift report data into Delta table
        
        Param
        -----
        target_profile_section: either NumTargetProfileSection or CatTargetProfileSection
        colmap: ColumnMapping saving column type information
        project_name: used as database name when saving data
        model_name: used as table_name when saving data
        """
        profile = Profile(sections=[target_profile_section()])
        profile.calculate(self.ref, self.prod, column_mapping=colmap)
        profile_js = js.loads(profile.json())
        profile_pyspark = spark.createDataFrame(pd.DataFrame(profile_js['num_target_drift']['data']['metrics']).reset_index())
        profile_pyspark = profile_pyspark.withColumn("date", F.current_date())
        
        output_table = f"{project_name}.{model_name}_label_drift"
        profile_pyspark.write.format("delta").mode("append").saveAsTable(output_table)
        print(f"label drift data written to {output_table}")

# COMMAND ----------

class ModMonDatasetDrift:
    def __init__(self, ref, prod):
        self.ref = ref
        self.prod = prod
        
    def report_dataset_drift(self, colmap, html_filepath=None, save_html=False):
        """
        create label drift report
        
        Param
        -----
        colmap: ColumnMapping saving column type information
        html_filepath: path to save report as html
        save_html: choose to save report as html
        """
        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(self.ref, self.prod, column_mapping=colmap)
        if save_html == True:
            dashboard.save(html_filepath)
        return dashboard

    def save_dataset_drift_data(self, colmap, project_name, model_name):
        """
        save label drift report data into Delta table
        
        Param
        -----
        colmap: ColumnMapping saving column type information
        project_name: used as database name when saving data
        model_name: used as table_name when saving data
        """
        from collections import namedtuple
        profile = Profile(sections=[DataDriftProfileSection()])
        profile.calculate(self.ref, self.prod, column_mapping=colmap)
        profile_js = js.loads(profile.json())
        
        DriftMetrics = namedtuple("DriftMetrics", "feature drift_detected drift_score feature_type stattest_name")
        feat_names = profile_js['data_drift']['data']['cat_feature_names'] + profile_js['data_drift']['data']['num_feature_names']
        datadrift_df = []
        drift_cols = ['drift_detected', 'drift_score', 'feature_type', 'stattest_name']
        for f in feat_names:
            data = {k:v for k,v in profile_js['data_drift']['data']['metrics'][f].items() if k in drift_cols}
            feature_drift_data = DriftMetrics(f, data['drift_detected'], data['drift_score'], data['feature_type'], data['stattest_name'])
            datadrift_df.append(feature_drift_data)
        
        profile_pyspark = spark.createDataFrame(pd.DataFrame(datadrift_df))
        profile_pyspark = profile_pyspark.withColumn("date", F.current_date())
        
        output_table = f"{project_name}.{model_name}_dataset_drift"
        profile_pyspark.write.format("delta").mode("append").saveAsTable(output_table)
        print(f"label drift data written to {output_table}")

# COMMAND ----------

class ModMonAggView():
    pass

# COMMAND ----------

class ModMonHistoricalView:
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
    
    def features_historical_drift():
        features_historical_drift = []
        experiment_batches = ["2022-01-01", "2022-02-01", "2022-03-01"]

        for batch in experiment_batches:
            drifts = detect_features_drift(raw_data[raw_data["date"]=="2022-01-01"], 
                           raw_data[raw_data["date"]==batch], 
                           column_mapping=data_columns)
    
        features_historical_drift.append([x[1] for x in drifts])
    
        features_historical_drift_frame = pd.DataFrame(features_historical_drift, 
                                               columns = data_columns.categorical_features+ data_columns.numerical_features)
        return features_historical_drift_frame

# COMMAND ----------

class retrain():
    pass
    
#     def agg_viz(self):
#         pass
    
#     def retrain(self):
#         """
#         take new data, retrain model, promote to staging if metric > previous model
#         """
