import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories, get_df
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.utils.featurize import save_matrix
import random
import numpy as np


STAGE = "Two" ## <<< change stage name 

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
    prepare_data_dir = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])    
    train_data_path = os.path.join(prepare_data_dir,artifacts["TRAIN_DATA"])
    test_data_path = os.path.join(prepare_data_dir,artifacts["TEST_DATA"])

    featurized_data_dir = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])    
    create_directories([featurized_data_dir])
    featurized_train_data_path = os.path.join(featurized_data_dir,artifacts["FEATURIZED_OUT_TRAIN"])
    featurized_test_data_path = os.path.join(featurized_data_dir,artifacts["FEATURIZED_OUT_TEST"])

    max_features = params['featurize'] ["max_params"]
    ngrams = params['featurize']['ngrams']

    df_train = get_df(train_data_path)

    train_words = np.array(df_train.text.str.lower().values.astype("U"))
    
    #generate feature matrix 
    bag_of_words = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1,ngrams)
    )
    bag_of_words.fit(train_words)
    train_words_binary_matrix =bag_of_words.transform(train_words)

    tfidf = TfidfVectorizer(smooth_idf=False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

    save_matrix(df_train, train_words_tfidf_matrix, featurized_train_data_path)
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