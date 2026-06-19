import pandas as pd
import numpy as np

import os
import shutil

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


# File Paths
file_path = "data/ratings.csv"
model_path = "models/als_models"

spark = SparkSession.builder \
        .appName("system-recommendation") \
        .getOrCreate()

spark = (
    SparkSession.builder
        .appName("system-recommendation")
        .getOrCreate()
)

spark.sparkContext.setLogLevel('ERROR')

spark_df = spark.read.csv(file_path, header=True, inferSchema=True)

(train_df, test_df) = spark_df.randomSplit([0.8, 0.2])

als_model = ALS(
    rank=10,
    maxIter=10,
    regParam=0.1,
    userCol="user_id",
    itemCol="item_id",
    ratingCol="rating",
    coldStartStrategy='drop'
)

model = als_model.fit(train_df)

test_predictions = model.transform(test_df)

evaluator = RegressionEvaluator(
    metricName='rmse',
    labelCol='rating',
    predictionCol='prediction'
)

rmse = evaluator.evaluate(test_predictions)
print(f"\nRMSE: {rmse:.4f}\n")

if os.path.exists(model_path):
    shutil.rmtree(model_path)

model.save(model_path)
print(f"\nModel saved.\n")






