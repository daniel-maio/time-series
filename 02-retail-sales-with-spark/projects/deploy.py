# ==================== Imports ====================

from pyspark.sql import SparkSession

from pyspark.sql.functions import col, year, month

from pyspark.sql import functions as F

from pyspark.ml import PipelineModel

# ==================== Configs ====================

spark = SparkSession.builder \
    .appName("LinearRegression-deploy") \
    .getOrCreate()

data_path = "/opt/spark/data/dataset.csv"

model_path = "/opt/spark/models/lr"

predictions_path = "/opt/spark/data/predictions"

model = PipelineModel.read().load(model_path)

# ==================== Predictions ====================

start_date = '2025-01-01'
end_date = '2025-06-01'

new_df = spark.sql(f"""
    SELECT explode(sequence(to_date('{start_date}'), to_date('{end_date}'), interval 1 month))
    AS date""")

new_df = new_df.withColumn(colName="year", col=year(col("date")))

new_df = new_df.withColumn(colName="month", col=month(col("date")))

new_df_predictions = model.transform(new_df)

new_df_predictions.select('date', 'prediction').show()

new_df_predictions.select('date', 'prediction').coalesce(1).write.csv(
    path=f"{predictions_path}/deploy",
    mode='overwrite',
    header=True)

spark.stop()