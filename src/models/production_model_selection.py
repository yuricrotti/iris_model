import joblib
import mlflow
import argparse
from pprint import pprint
from train_model import read_params
from mlflow.tracking import MlflowClient

def log_production_model(config_path):
    # Read params config
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"] 
    model_name = mlflow_config["registered_model_name"]
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Get the experiment name
    EXPERIMENT_NAME = mlflow_config['experiment_name']

    # Set the tracking URI
    mlflow.set_tracking_uri(remote_server_uri)

    # Create a MlflowClient
    client = MlflowClient()

    # Retrieve Experiment information
    EXPERIMENT_ID = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    # Search for all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=EXPERIMENT_ID)

    # Get the run id with the highest accuracy
    max_accuracy = max(runs["metrics.accuracy"])
    max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]

    print("Max Accuracy: ", max_accuracy)
    print("Max Accuracy Run ID: ", max_accuracy_run_id)
    
    # Get the experiment
    client.get_experiment(EXPERIMENT_ID)
    

    # for each model version in the model name
    for mv in client.search_model_versions(f"name='{model_name}'"):
        # Convert to dictionary model version
        mv = dict(mv)
        # If the run id is the same as the max accuracy run id, then log the model as production
        if mv["run_id"] == max_accuracy_run_id:
            # Get the current version
            current_version = mv["version"]
            # Get the model path
            logged_model = mv["source"]
            # 
            pprint(mv, indent=4)
            # Transition the model to production
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
        # Else, transition the model to staging
        else:
            # Get the current version
            current_version = mv["version"]
            # Transition the model to staging
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )
    # Load the model
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    # Save the model
    joblib.dump(loaded_model, model_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)