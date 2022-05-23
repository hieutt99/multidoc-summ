import ruamel.yaml as ruamel_yaml
import os 

def load_config_yaml(path):
    if os.path.exists(path):
        with open(path, 'r') as fp:
            yaml_str = fp.read()
        data = ruamel_yaml.safe_load(yaml_str)
        return data
    else:
        raise Exception("Error loading yaml")