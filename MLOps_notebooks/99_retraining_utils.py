# Databricks notebook source
# code repurposed from Wilson, B. (2022). Machine Learning Engineering in Action. Manning.

@dataclass
class Registry:
    model_name: str
    production_version: int
    updated: bool
    training_time: str
    
class RegistryStructure:
    def __init__(self, data):
        self.data = data
    def generate_row(self):
        spark_df = spark.createDataFrame(pd.DataFrame([vars(self.data)]))
        return (spark_df.withColumn("training_time", F.to_timestamp(F.col("training_time")))
            .withColumn("production_version", 
F.col("production_version").cast("long")))
        
class RegistryLogging:
    def __init__(self, 
               database, 
               table, 
               delta_location, 
               model_name, 
               production_version, 
               updated):
        self.database = database
        self.table = table
        self.delta_location = delta_location
        self.entry_data = Registry(model_name, 
                               production_version, 
                               updated, 
                               self._get_time())
    @classmethod
    def _get_time(self):
        return datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    def _check_exists(self):
        return spark._jsparkSession.catalog().tableExists(self.database, self.table)
    def write_entry(self):
        log_row = RegistryStructure(self.entry_data).generate_row()
        log_row.write.format("delta").mode("append").save(self.delta_location)
        if not self._check_exists():
            spark.sql(f"""CREATE TABLE IF NOT EXISTS 
            {self.database}.{self.table} USING DELTA LOCATION'{self.delta_location}';""")

# COMMAND ----------

class ModelRegistration:
    def __init__(self, experiment_name, experiment_title, model_name, metric,
               direction):
        self.experiment_name = experiment_name
        self.experiment_title = experiment_title
        self.model_name = model_name
        self.metric = metric
        self.direction = direction
        self.client = MlflowClient()
        self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    def _get_best_run_info(self, key):
        run_data = mlflow.search_runs(
          self.experiment_id, 
          order_by=[f"metrics.{self.metric} {self.direction}"])
        return run_data.head(1)[key].values[0]
    def _get_registered_status(self):
        return self.client.get_registered_model(name=self.experiment_title)
    def _get_current_prod(self):
        return ([x.run_id for x in self._get_registered_status().latest_versions if x.current_stage == "Production"][0])
    def _get_prod_version(self):
        return int([x.version for x in self._get_registered_status().latest_versions
             if x.current_stage == "Production"][0])
    def _get_metric(self, run_id):
        return mlflow.get_run(run_id).data.metrics.get(self.metric)
    def _find_best(self):
        try: 
            current_prod_id = self._get_current_prod()
            prod_metric = self._get_metric(current_prod_id)
        except mlflow.exceptions.RestException:
            current_prod_id = -1
            prod_metric = 1e7
        best_id = self._get_best_run_info('run_id')
        best_metric = self._get_metric(best_id)
        if self.direction == "ASC":
            if prod_metric < best_metric:
                return current_prod_id
            else:
                return best_id
        else:
            if prod_metric > best_metric:
                return current_prod_id
            else:
                return best_id
    def _generate_artifact_path(self, run_id):
        return f"runs:/{run_id}/{self.model_name}"
    def register_best(self, registration_message, logging_location, log_db, log_table):
        best_id = self._find_best()
        try:
            current_prod = self._get_current_prod()
            current_prod_version = self._get_prod_version()
        except mlflow.exceptions.RestException:
            current_prod = -1
            current_prod_version = -1
        updated = current_prod != best_id
        if updated:
            register_new = mlflow.register_model(self._generate_artifact_path(best_id),
                                   self.experiment_title)
            self.client.update_registered_model(name=register_new.name, 
                                          description="Latest model")
            self.client.update_model_version(name=register_new.name, 
                                       version=register_new.version, 
                                       description=registration_message)
            self.client.transition_model_version_stage(name=register_new.name, 
                                                 version=register_new.version,
                                                 stage="Production")
            if current_prod_version > 0:
                self.client.transition_model_version_stage(name=register_new.name, 
                                                         version=current_prod_version,
                                                         stage="Archived")
            RegistryLogging(log_db, 
                    log_table, 
                    logging_location, 
                    self.experiment_title,  
                    int(register_new.version), 
                    updated).write_entry()
            return "upgraded prod"
        else:
            RegistryLogging(log_db, 
                log_table, 
                logging_location, 
                self.experiment_title, 
                int(current_prod_version), 
                updated).write_entry()
            return "no change"
    def get_model_as_udf(self):
        prod_id = self._get_current_prod()
        artifact_uri = self._generate_artifact_path(prod_id)
        return mlflow.pyfunc.spark_udf(spark, model_uri=artifact_uri)

# COMMAND ----------


