import os
import yaml
import pandas as pd
import argparse
from pkgutil import get_data
from get_data import get_data,read_params
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import json
import joblib
import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error



def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    model_dir = config["model_dir"]
    target = config["base"]["target_col"]
    max_depth = config["estimators"]["RandomForestRegressor"]["params"]["max_depth"]
    random_state = config["base"]["random_state"]
    train = pd.read_csv(train_data_path,sep=",")
    test = pd.read_csv(test_data_path,sep=",")

    train_x = train.drop(target,axis=1)
    test_x = test.drop(target,axis=1)
    train_y = train[target]
    test_y = test[target]

    ########################################################################################
    rf = RandomForestRegressor(max_depth=max_depth,random_state=random_state)
    rf.fit(train_x,train_y)
    prediction = rf.predict(test_x)

    (rmse,mae,r2) = eval_metrics(test_y,prediction)

    # print("RMSE%s",rmse)
    # print("MAE%s",mae)
    # print("r2%s",r2)

    ########################################################################################

    score_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(score_file,"w") as f:
        scores = {
            "RMSE":rmse,
            "MAE":mae,
            "R2_SCORE":r2
        }
        json.dump(scores,f,indent=4)

    with open(params_file,"w") as f:
        params = {
            "max_depth":max_depth
        }
        json.dump(params,f,indent=4)

    os.makedirs(model_dir,exist_ok=True)
    model_path  = os.path.join(model_dir,"model.joblib")
    joblib.dump(rf,model_path)
    









if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path = parsed_args.config)
