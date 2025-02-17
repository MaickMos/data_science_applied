import pandas as pd
import numpy as np
import os
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


def predict_probability(user_id,order_id,database):
    

    # Evaluación
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    # Guardar modelo
    model_path = f"models/logistic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")

    return model

def main():
    filepath = "data/orders_data.csv" 
    df = load_data(filepath)
    
    if df is not None:
        df = preprocess_data(df)
        X = df.drop(columns=["outcome"])
        y = df["outcome"]
        trained_model = train_model(X, y)

if __name__ == "__main__":
    main()