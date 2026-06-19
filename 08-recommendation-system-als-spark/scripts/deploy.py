import socket
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

model_path = "models/als_models"

spark = (
    SparkSession.builder
    .appName("session")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("Error")

model = ALSModel.load(model_path)

def recomend(user_id: int, num_recs: int):
    
    users_df = spark.createDataFrame(data=[{'user_id': user_id}])

    user_recs = model.recommendForUserSubset(users_df, num_recs)

    recs = user_recs.collect()

    if recs:

        recommendations = recs[0].recommendations

        recommendations = [{"item_id":row["item_id"], "rating":row["rating"]} for row in recommendations]

        print ({"user_id": user_id, "recommendations": recommendations})

    else:

        print(f"No recommendations were found for this user.")   
    
recomend(1, 5)