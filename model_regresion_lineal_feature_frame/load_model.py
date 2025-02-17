import pandas as pd
import numpy as np
import Path 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        required_columns = {"user_order_seq","ordered_before","abandoned_before","active_snoozed","set_as_regular","normalised_price","discount_pct","global_popularity","count_adults","count_children","count_babies","count_pets","people_ex_baby","days_since_purchase_variant_id","avg_days_to_buy_variant_id","std_days_to_buy_variant_id","days_since_purchase_product_type","avg_days_to_buy_product_type","std_days_to_buy_product_type"}
        if not required_columns.issubset(df.columns):
            raise ValueError("Missing required columns in dataset!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):

    return df


def predict_probability(
        user_id: int,
        order_id: int,
        database_path: str,
        ):
    folder_path = Path(database_path)
    feature_frame = load_data(folder_path)

    data =preprocess_data(feature_frame[
        feature_frame["user_id"]==user_id and feature_frame["order_id"]==order_id]
        [["ordered_before","global_popularity","abandoned_before"]]
        )

    model = joblib.load("modelo_push_notifications.pkl")

    y_pred = model.predict(data)

    return y_pred

def main():
    predict_probability

if __name__ == "__main__":
    main()