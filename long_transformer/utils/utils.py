import ruamel.yaml as yaml
import os 

def load_config_yaml(path):
    if os.path.exists(path):
        with open(path, 'r') as fp:
            yaml_str = fp.read()
        data = yaml.safe_load(yaml_str)
    else:
        raise Exception("Error loading yaml")