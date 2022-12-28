import os
import tempfile
import pandas as pd
# import sys
# sys.path.append("/src/data")
from src.data.load_data import load_raw_data
#from load_data import load_raw_data

def test_load_raw_data():
    # Create a temporary directory to store the raw data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fake external data file
        external_data_path = os.path.join(temp_dir, "external.csv")
        pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}).to_csv(external_data_path, index=False)

        # Create a fake config file
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(f"""
            external_data_config:
                external_data_csv: {external_data_path}
            raw_data_config:
                raw_data_csv: {os.path.join(temp_dir, "raw.csv")}
                model_var: ["col1"]
            """)

        # Run the load_raw_data function with the fake config file
        load_raw_data(config_path)

        # Read the generated raw data file and check that it has the correct columns
        raw_data = pd.read_csv(os.path.join(temp_dir, "raw.csv"))
        assert set(raw_data.columns) == {"col1"}