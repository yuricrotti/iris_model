import yaml
import argparse
import numpy as np
import pandas as pd

def read_params(config_path: str) -> str:
    """Function to read params

    Parameters
    ----------
    config_path : str
        path config file

    Returns
    -------
    str
        config
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def load_data(data_path: str,model_var: list)-> pd.DataFrame:
    """ Function to load csv dataset from given path

    Parameters
    ----------
    data_path : str
        path dataset
    model_var : list
        list of vars used in the model

    Returns
    -------
    pd.Dataframe
        dataframe
    """

    dataframe = pd.read_csv(data_path, sep=",", encoding='utf-8')
    dataframe = dataframe[model_var]
    return dataframe

def load_raw_data(config_path: str):
    """Function to load data from external location(data/external)
       to the raw folder(data/raw) with train and teting dataset 

    Parameters
    ----------
    config_path : str
        Path of config file
    """
    config=read_params(config_path)
    external_data_path=config["external_data_config"]["external_data_csv"]
    raw_data_path=config["raw_data_config"]["raw_data_csv"]
    model_var=config["raw_data_config"]["model_var"]

    dataframe = load_data(external_data_path,model_var)
    dataframe.to_csv(raw_data_path,index=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_raw_data(config_path=parsed_args.config)
