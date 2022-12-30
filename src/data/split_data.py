import os
import argparse
import pandas as pd
from load_data import read_params
from sklearn.model_selection import train_test_split

def split_data(dataframe : pd.DataFrame,
               train_data_path: str,
               test_data_path: str,
               split_ratio: float,
               random_state:int):
    """Function to split dataframe

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataframe to split
    train_data_path : str
        train data path
    test_data_path : str
        test data path
    split_ratio : float
        ratio of split
    random_state : int
        random state
    """
    # Split the data
    train, test = train_test_split(dataframe,
                                   test_size = split_ratio,
                                   random_state=random_state)
    # Save the data
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")

def split_and_saved_data(config_path: str):
    """Function to split the train dataset(data/raw) and save it in the data/processed folder

    Parameters
    ----------
    config_path : str
        config path
    """
    # read the config file
    config = read_params(config_path)
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"] 
    train_data_path = config["processed_data_config"]["train_data_csv"]
    split_ratio = config["raw_data_config"]["train_test_split_ratio"]
    random_state = config["raw_data_config"]["random_state"]

    # read the raw data
    raw_df=pd.read_csv(raw_data_path)

    # split the data
    split_data(raw_df,train_data_path,test_data_path,split_ratio,random_state)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(config_path=parsed_args.config)
