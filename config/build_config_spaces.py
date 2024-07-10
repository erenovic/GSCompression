
import argparse
import copy
import os
from pathlib import Path
from typing import Dict

import yaml


class ConfigReader:
    def __init__(self, config_file: str):
        self.config_file_path = Path(config_file)
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
    
    @property
    def dataset_config(self):
        dataset_args = argparse.Namespace(**self.config.get('dataset', {}))
        return dataset_args
    
    @property
    def pipeline_config(self):
        pipeline_args = argparse.Namespace(**self.config.get('pipeline', {}))
        return pipeline_args
    
    @property
    def optimization_config(self):
        optimizer_args = argparse.Namespace(**self.config.get('optimization', {}))
        return optimizer_args
    
    def append_args_to_config_and_save(self, args, model_path: str):
        new_config: Dict = copy.deepcopy(self.config)
        new_config.update({k: v for k, v in vars(args).items() if v is not None})

        # Serialize the updated YAML config to a new file
        with open(os.path.join(model_path, "configuration.yaml"), 'w') as f:
            yaml.safe_dump(new_config, f, default_flow_style=False)
    
    def modify_using_args(self, args):
        # Update self.config with command-line argument values
        for key in self.config:
            for k in self.config[key]:
                if hasattr(args, k) and (getattr(args, k) is not None):
                    self.config[key][k] = getattr(args, k)

    # Function to concatenate namespaces
    @classmethod
    def concatenate_namespaces(cls, *namespaces):
        combined_dict = {}
        for ns in namespaces:
            combined_dict.update(vars(ns))
        return argparse.Namespace(**combined_dict)
        