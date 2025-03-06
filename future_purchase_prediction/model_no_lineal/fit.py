import json
import pickle
import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

def load_data(
    file_path: str
    ):
    try:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        df = pd.read_csv(file_path)


        required_columns = {"user_order_seq","ordered_before","abandoned_before","active_snoozed","set_as_regular","normalised_price","discount_pct","global_popularity","count_adults","count_children","count_babies","count_pets","people_ex_baby","days_since_purchase_variant_id","avg_days_to_buy_variant_id","std_days_to_buy_variant_id","days_since_purchase_product_type","avg_days_to_buy_product_type","std_days_to_buy_product_type"}

        if not required_columns.issubset(df.columns):
            raise ValueError("Missing required columns in dataset!")
        return df
        
    except FileNotFoundError as e:
        print(f"Error not found: {file_path}")

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def Standardization(
        x_train: pd.DataFrame
        ):
    scaler = StandardScaler()
    x_train_columns = x_train.columns
    x_train = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train, columns=x_train_columns)
    return x_train

def slip_data_set(df,y):
    x_train = df.iloc[:int(len(df)*0.7)]
    x_test = df.iloc[int(len(df)*0.7):int(len(df)*0.9)]
    x_valid = df.iloc[int(len(df)*0.9):]

    y_train = y.iloc[:int(len(y)*0.7)]
    y_test = y.iloc[int(len(y)*0.7):int(len(y)*0.9)]
    y_valid = y.iloc[int(len(y)*0.9):]

    print(f"Train: {len(x_train)} and {len(y_train)} ")
    print(f"Valid: {len(x_test)} and {len(y_test)}")
    print(f"Test: {len(x_valid)} and {len(y_valid)}")

    x_train = Standardization(x_train)
    return x_train, x_test, x_valid, y_train, y_test, y_valid

def preprocess_data(
        feature_frame:pd.DataFrame,
        ):

    size_of_order = feature_frame.groupby("order_id").outcome.sum()
    size_of_order = size_of_order[size_of_order>=5]
    feature_frame = feature_frame[feature_frame["order_id"].isin(size_of_order.index)]
    feature_frame = (feature_frame.sort_values("order_date",ascending=True).reset_index()).drop("index",axis = 1)


    
    xf = feature_frame[["ordered_before","global_popularity","abandoned_before"]]

    x_train, x_test, x_valid, y_train, y_test, y_valid = slip_data_set(xf,feature_frame["outcome"])
    return x_train, y_train


def save_model_with_date(model):
    model_name = f"push_{datetime.now().strftime('%Y_%m_%d')}.pkl"
    model_path = os.path.join("D:/Users/maick/Desktop/Codigos/Data-science-applied/future_purchase_prediction/model_no_lineal/models", model_name)
    os.makedirs("models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path

def handler_fit(event, _):
    model_params = event.get("model_parametrisation", {})

    folder_path = "D:/Users/maick/Desktop/Codigos/zrive-ds/data/box_builder_dataset/feature_frame.csv"
    data = load_data(folder_path)

    x_train, y_train = preprocess_data(data)

    model = LogisticRegression(**model_params)
    model.fit(x_train, y_train)
    
    model_path = save_model_with_date(model)
    
    return {"statusCode": 200, "body": json.dumps({"model_path": model_path})}

def main():
    event = {
        "model_parametrisation" :{"penalty":"l2", "solver":"liblinear"}
            }
    model = handler_fit(event, 4)
    print(model["body"])

if __name__ == "__main__":
    main()
