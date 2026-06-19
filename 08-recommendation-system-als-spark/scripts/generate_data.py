import numpy as np
import pandas as pd

rng = np.random.default_rng()

file_path = "data/ratings.csv"


def generate_data(num_users, num_items, start_date, end_date, data_size):

    user_id = rng.integers(1, num_users, data_size, endpoint=True)
    
    item_id = rng.integers(1, num_items, data_size, endpoint=True)

    rating = np.round(rng.uniform(1, 5, data_size), 2)

    # Date Column

    time_between_dates = (end_date - start_date).days

    random_days = rng.integers(1, time_between_dates, data_size, endpoint=True)

    dates = start_date + pd.to_timedelta(random_days, unit='D')

    # Create DataFrame

    df = pd.DataFrame(
        data=
            {   
                "date": dates,
                "user_id": user_id,
                "item_id": item_id,
                "rating": rating
            }
    )

    # Sort by user_id
    
    df.sort_values(by="user_id", inplace=True)
    
    return df

num_users = 100
num_items = 300
start_date = pd.to_datetime("2022-01-01")
end_date = pd.to_datetime("2024-12-31")
data_size = 1_000

df = generate_data(num_users, num_items, start_date, end_date, data_size)

df.to_csv(file_path, header=True, index=False)