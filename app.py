from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import yaml
import joblib 

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, template_folder=template_dir) #Initialize the flask App

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def form_response(dict_request):
    
    config = read_params(params_path)
    model_dir_path = config["model_webapp_dir"]
    data = dict_request.values()
    data = [list(map(float, data))]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    response = prediction
    return response


@app.route("/", methods=['GET', 'POST'])
def index():

    if request.method == "POST":
        if request.form:
            dict_req = dict(request.form)
            response = form_response(dict_req)
            return render_template("index.html", prediction_text=response)
    else:

        return render_template("index.html",prediction_text="")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)