import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

STAGE = "THREE" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    featurized_data_dir = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])    
    featurized_train_data_path = os.path.join(featurized_data_dir,artifacts["FEATURIZED_OUT_TRAIN"])
    
    model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])
    create_directories([model_dir_path])
    model_path = os.path.join(model_dir_path, artifacts['MODEL_NAME'])

    matrix = joblib.load(featurized_train_data_path)
    labels = np.squeeze(matrix[:,1].toarray())
    X = matrix[:,2:]
    
    seed = params["train"]["seed"]
    n_est = params ["train"]["n_est"]
    min_splits = params ["train"]["min_split"]

    model = RandomForestClassifier(
        n_estimators= n_est,
        min_samples_split= min_splits,
        n_jobs= 2,
        random_state= seed
    )

    model.fit(X, labels)
    joblib.dump(model, model_path)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e