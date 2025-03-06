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

def preprocess_data(
        df:pd.DataFrame,
        user_id:list | int,
        variant_id:list | int
        ):

    dataframe = pd.DataFrame()

    for i in range(len(user_id)):
        df_filter = df[(df["user_id"] == user_id[i]) & (df["variant_id"] == variant_id[i])]

        if not df_filter.empty:           
            dataframe[f"{i+1}"] = df_filter[["ordered_before","global_popularity","abandoned_before"]].iloc[0]
        else:
            print(f"Not exit data for user:{user_id}  or product:{variant_id}")
            dataframe[f"{i+1}"] = [None, None, None] 

    if dataframe.empty:
        print(f"No user or product in database")
        return None
    
    return dataframe.T

def load_model_from_path(
        model: str,#file_path
        scaler: str#file_path
    ):
    try:
        model = joblib.load(model)
        scaler = joblib.load(scaler)
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model, scaler

def Standardization(
        scaler: object,#StandardScaler
        x_train: pd.DataFrame
        ):
    
    x_train_columns = x_train.columns
    x_train = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train, columns=x_train_columns)
    return x_train

def predict_probability(
        user_id: list | int,
        variant_id: list | int,
        database_path: str,
        path_model:str,
        path_scaler:str
        ):
    
    folder_path = Path(database_path)
    data = load_data(folder_path)

    data = preprocess_data(data,user_id,variant_id)
    

    model, scaler = load_model_from_path(path_model,path_scaler)
    x_data = Standardization(scaler,data)
    y_pred = model.predict(x_data)


    return y_pred

def main():
    user_id = [3771764834436,  3781248352388]
    variant_id = [34137389072516, 34037942452356]
    database_path = "C:/Users/AULA04/Desktop/Codes/zrive-ds/data/box_builder_dataset/feature_frame.csv"
    path_model = "C:/Users/AULA04/Desktop/Codes/data_science_applied/model_regresion_lineal_feature_frame/modelo_push_notifications_1.pkl"
    path_scaler = "C:/Users/AULA04/Desktop/Codes/data_science_applied/model_regresion_lineal_feature_frame/scaler.pkl"
    prediction = predict_probability(user_id, variant_id, database_path,path_model,path_scaler)

    print(prediction)

if __name__ == "__main__":
    main()