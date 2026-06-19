# ==================== Imports ====================

from pyspark.sql import SparkSession

from pyspark.sql.functions import col, year, month
from pyspark.sql.types import DateType

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml import Pipeline

# ==================== Configs ====================

spark = SparkSession.builder \
    .appName("LinearRegression-train") \
    .getOrCreate()

data_path = "/opt/spark/data/dataset.csv"

model_path = "/opt/spark/models/lr"

predictions_path = "/opt/spark/data/predictions"

# ==================== Pre-Processing ====================

df = spark.read.csv(path=data_path, header=True, inferSchema=True)

df = df.withColumn(colName="Date", col=col("Date").cast(dataType=DateType()))

df = df.withColumnsRenamed(colsMap={"Date": "date", "Sales": "sales"})

df = df.withColumn(colName="year", col=year(col("date")))

df = df.withColumn(colName="month", col=month(col("date")))

feature_assembler = VectorAssembler(inputCols=["year", "month"], outputCol="features")

df_train, df_valid = df.randomSplit([0.7, 0.3])

# ==================== Linear Regression Model ====================

lr = LinearRegression(featuresCol="features", labelCol="sales")

pipeline = Pipeline(stages=[feature_assembler, lr])

model = pipeline.fit(dataset=df_train)

df_pred = model.transform(dataset=df_valid)

# ==================== Evaluation ====================

lr_ev = RegressionEvaluator(predictionCol="prediction", labelCol="sales", metricName="rmse")

rmse = lr_ev.evaluate(dataset=df_pred)
print(f"RMSE on the validation set: {rmse:.6f}")

# ==================== Save ====================

model.write().overwrite().save(model_path)

df_pred.select(["date", "sales", "prediction"]).write.csv(path=f"{predictions_path}/valid", header=True, mode='overwrite')

# ==================== Stop Spark Session ====================

spark.stop()