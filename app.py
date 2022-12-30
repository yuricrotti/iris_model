import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import yaml
import joblib

WEBAPP_ROOT = "webapp"
PARAMS_PATH = "params.yaml"


static_dir = os.path.join(WEBAPP_ROOT, "static")
template_dir = os.path.join(WEBAPP_ROOT, "templates")

app = Flask(__name__, template_folder=template_dir)  # Initialize the flask App


def read_params(config_path: str):
    """Function to read the parameters from the yaml file

    Parameters
    ----------
    config_path : str
        Path to the yaml file

    Returns
    -------
    config : dict
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def form_response(dict_request: dict):
    """
    Function to get the response from the model for the form request

    Parameters
    ----------
    dict_request : dict
        Dictionary of the form request

    Returns
    -------
    response : str"""

    config = read_params(PARAMS_PATH)
    model_dir_path = config["model_webapp_dir"]
    data = dict_request.values()
    data = [list(map(float, data))]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    response = prediction
    return response


@app.route("/", methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    if request.method == "POST":
        # if request.form:
        if request.form:
            # Convert MultiDict to regular dict
            dict_req = dict(request.form)
            # Get the response using our form response function
            response = form_response(dict_req)
            # Render the form again, but add in the prediction.
            return render_template("index.html", prediction_text=response)
    else:
        return render_template("index.html", prediction_text="")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
