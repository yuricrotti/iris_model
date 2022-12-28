import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score,recall_score,accuracy_score, 
                            precision_score,confusion_matrix,classification_report)
from sklearn.model_selection import GridSearchCV

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def accuracymeasures(y_test,predictions,avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score

def train_and_evaluate(config_path):
    """_summary_

    Parameters
    ----------
    config_path : _type_
        _description_
    """
    # read config file
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_x, train_y = get_feat_and_target(train,target)
    test_x, test_y = get_feat_and_target(test,target)

    ################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        # Define parameter grid for grid search
        param_grid = {'n_estimators': [1, 10, 50, 100, 200],
                      'max_samples': [0.1, 0.5, 1.0, 1, 5, 10]}

        model = RandomForestClassifier()

        # Create the grid search object
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

        # Fit the grid search to the data
        grid_search.fit(train_x, train_y)

        # Save the best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Use best model to predict on test data
        y_pred = best_model.predict(test_x)

        # Calculate metrics
        accuracy, precision, recall, f1score = accuracymeasures(test_y,y_pred,'weighted')

        # Log the parameters
        mlflow.log_param("params", best_params)

        # Log the metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)

        # 
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                best_model,
                "model",
                registered_model_name=mlflow_config["registered_model_name"])

        # Local model registry
        else:
            mlflow.sklearn.load_model(best_model, "model")

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)