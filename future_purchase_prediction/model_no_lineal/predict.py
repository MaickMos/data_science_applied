import os
import pickle
import pandas as pd
import json
from pathlib import Path

def handler_predict(event, _):
    try:
        model_files = sorted(os.listdir(event["model_path"]), reverse=True)

        if not model_files:
            raise FileNotFoundError("No models trained.")
        #Select the lastest
        print(model_files[0])
        model_path = os.path.join(event["model_path"], model_files[0])
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        

        data_to_predict = pd.DataFrame.from_dict(json.loads(event["users"]), orient="index")
        
        predictions = model.predict_proba(data_to_predict)[:, 1]
        result = {}
        for user_id, pred in zip(data_to_predict.index,predictions):
            result[user_id] = float(pred)
        
        return {"statusCode": 200, "body": json.dumps({"prediction": result})}
    
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

def main():
    json_path = Path("D:/Users/maick/Desktop/Codigos/Data-science-applied/future_purchase_prediction/model_no_lineal/filtered_users.json")

    with open(json_path, "r", encoding="utf-8") as f:
        user_data = json.load(f)

    event = {
        "model_path": "D:/Users/maick/Desktop/Codigos/Data-science-applied/future_purchase_prediction/model_no_lineal/models",
        "users": json.dumps(user_data)  # Convertir el diccionario a cadena JSON
    }
    
    model = handler_predict(event, None)

    print(model)

if __name__ == "__main__":
    main()

